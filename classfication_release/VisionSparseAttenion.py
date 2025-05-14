import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from typing import Tuple


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        B , heads , region , w2 , c =kv.size()
        # b h r k
        topk = r_idx.size(-1)
        # b h r k w2 c
        topk_kv = torch.gather(
            kv.view(B,heads,region,1,w2,c).expand(-1,-1,-1,region,-1,-1),
            dim=3,
            index=r_idx.view(B,heads,region,topk,1,1).expand(-1,-1,-1,-1,w2,c)
            )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(B , heads , region , topk , 1 ,1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        else: #'none'
            topk_kv = topk_kv # do nothing
        return topk_kv

        # # select kv according to routing index
        # n, p2, w2, c_kv = kv.size()
        # topk = r_idx.size(-1)
        # # print(r_idx.size(), r_weight.size())
        # # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel? 
        # topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
        #                         dim=2,
        #                         index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
        #                        )

        # if self.mul_weight == 'soft':
        #     topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        # elif self.mul_weight == 'hard':
        #     raise NotImplementedError('differentiable hard routing TBA')
        # # else: #'none'
        # #     topk_kv = topk_kv # do nothing

        # return topk_kv


class TopkRouting(nn.Module): 
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
            
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index
        

def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class VisionSparseAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim_head: int,
        heads: int,
        window_size: int,   # sliding window 的边长（建议为奇数，便于中心对称）
        patch_size: int,    # token compression 时的 patch 大小（假设 H,W 能被整除）
        num_selected: int,  # fine selection 分支中每个查询选择的候选区域数量
        use_pos_emb: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim_head = dim_head
        self.heads = heads
        self.total_dim = dim_head * heads
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_selected = num_selected

        self.scale = dim_head ** -0.5

        # 对输入 token（展平后的像素）进行 qkv 线性映射
        # self.to_qkv = nn.Linear(in_channels, self.total_dim * 3, bias=False)
        self.to_qkv = nn.Conv2d(in_channels=in_channels , out_channels= 3*self.total_dim,kernel_size= 7 , stride= 1 , padding=3)
        # token compression 分支：利用卷积将图像分块压缩
        self.compressConV = nn.Conv2d(in_channels=3*self.total_dim , out_channels= 3*self.total_dim ,kernel_size=patch_size , stride=patch_size )
        
        self.router = TopkRouting(qk_dim=self.total_dim,
                                  qk_scale=self.scale,
                                  topk=self.num_selected,
                                  diff_routing=False,
                                  param_routing=False)
        
        self.kv_gather = KVGather(mul_weight="none")
        # 输出投影
        self.out_proj = nn.Linear(self.total_dim, in_channels)

        # # 可学习的位置编码（这里用一个 3x3 卷积作为简单的 2D 位置编码器）
        # if use_pos_emb:
        #     self.pos_emb = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # else:
        #     self.pos_emb = None

        # 为 compression 分支准备一个投影模块，将压缩后的 token 映射到 total_dim 维度后拆分为 k 和 v
        self.comp_proj = nn.Linear(in_channels, self.total_dim * 2, bias=False)

    def forward(self, x ):
        """
        x: [B, C, H, W]
        输出: [B, C, H, W]
        """
        B, C, H, W = x.shape


        # # 加入位置编码（可选）
        # if self.pos_emb is not None:
        #     pos = self.pos_emb(x)  # [B, C, H, W]
        #     x = x + pos
        
        # # 将图像展平为 token 序列 [B, H*W, C]
        # tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]，其中 N = H*W
        
        # 计算 q, k, v 对应高分辨率的 fine token
        qkv = self.to_qkv(x)  # [B,3*dim , H , W]

        qfine , kfine , vfine = qkv.chunk(3,dim= 1) #dim = b c h w
        compressQKV = self.compressConV(qkv)  # [ B , 3*dim , Hc , Wc]
        comQ , comK , comV = torch.chunk(compressQKV , chunks=3 , dim=1)
        Bc, Cc, Hc, Wc = comQ.shape
        
        
        #分头：
        qfine_heads = rearrange(qfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        kfine_heads = rearrange(kfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        vfine_heads = rearrange(vfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        
        # (sin_fine , cos_fine) , mask_fine = rel_pos_fine
        # # 在这里刚好对qfine 和kfine 做位置编码
        # qfine_heads = theta_shift(qfine_heads , sin_fine , cos_fine)
        # kfine_heads = theta_shift(kfine_heads , sin_fine , cos_fine)


        # q, k, v = qkv.chunk(3, dim=-1)
        # # 分头：转换为 [B, heads, N, dim_head]
        # q = q.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        # k = k.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        # v = v.view(B, -1, self.heads, self.dim_head).transpose(1, 2)


        # 这个前置需要什么？需要kvin vvin 以及 qfine 
        # ========= 分支 1：Token Compression =========
        # 利用卷积将图像压缩成较低分辨率的区域表示
        # comp = self.compress_conv(x)  # [B, C, Hc, Wc]，其中 Hc = H/patch_size, Wc = W/patch_size
  
        num_regions = Hc * Wc
        # comp_tokens = comp.flatten(2).transpose(1, 2)  # [B, num_regions, C]
        # # 线性映射后拆分为 compressed k 和 v
        # comp_proj = self.comp_proj(comp_tokens)  # [B, num_regions, 2 * total_dim]
        # comp_k, comp_v = comp_proj.chunk(2, dim=-1)
        # TODO: 两个heads需要不一样吗？
        comQ = rearrange(comQ , "b (heads c) h w -> b heads (h w) c",heads=self.heads)
        comK = rearrange(comK , "b (heads c) h w -> b heads (h w) c",heads=self.heads)
        comV = rearrange(comV , "b (heads c) h w -> b heads (h w) c",heads=self.heads)
        # (sin_comp , cos_comp) , mask_comp = rel_pos_comp
        # comQ = theta_shift(comQ , sin_comp , cos_comp)
        # comK = theta_shift(comK , sin_comp , cos_comp)
        # # 分头：调整为 [B, heads, num_regions, dim_head]
        # comp_k = comp_k.view(B, num_regions, self.heads, self.dim_head).transpose(1, 2)
        # comp_v = comp_v.view(B, num_regions, self.heads, self.dim_head).transpose(1, 2)
        # 利用 q 与 comp_k 计算区域相似度（每个像素对各区域的响应）
        # q : [b heads n dh] comk : b heads N dh
        sim_comp = torch.matmul(qfine_heads.view(B , self.heads , H*W , self.dim_head), comK.transpose(-1, -2)) * self.scale  # [B, heads, N, num_regions]
        attn_comp = sim_comp.softmax(dim=-1)
        out_comp = torch.matmul(attn_comp, comV)  # [B, heads, n, dim_head]



        # # ========= 分支 2：Fine Token Selection =========
        # shape为：B p^2 k ( P2 就是块的数量)
        r_weight, r_idx = self.router(comQ, comK) # both are (n, p^2, topk) tensors
        # 维度：B h r k w2 c
        kpatch = rearrange(qfine,'b (heads c) (j h) (i w)  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads)
        vpatch = rearrange(vfine,'b (heads c) (j h) (i w)  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads)
        # # b h r k w2 c
        gatherK = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kpatch) 
        gatherV = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=vpatch)
        
        # r: regions c: dim_heads w2: patchsize^2 k : topk 
        # 已经做好转置了
        # 这个维度还得再确认一下，为什么region要提取出来
        
        # gatherK = rearrange(gatherK,'b r k w2 (heads c) -> (b r) heads c (k w2)',heads=self.heads)
        # gatherV = rearrange(gatherV,'b r k w2 (heads c) -> (b r) heads c (k w2)',heads=self.heads) 
        # br h kw2 c 
        gatherK = rearrange(gatherK,'b h r k w2 c -> (b r) h c (k w2)')
        gatherV = rearrange(gatherV,'b h r k w2 c -> (b r) h (k w2) c')  
        # br h w2 c
        qpatch = rearrange(qfine,'b (heads c) (j h) (i w)  -> (b j i) heads (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads)

        # 说不定q也需要转换一下，把region提取出来
        attn_fine = ((qpatch * self.scale) @ gatherK).softmax(-1)
        # br h w2 c
        out_fine = attn_fine @ gatherV
        #TODO: 这里的维度还需要确认一下
        out_fine = rearrange(out_fine,'(b r) h w2 c -> b h (r w2)  c',r=Hc*Wc)

        
        # k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        # 所以前置需要： qwin  kwin qfine kfine vfind
        # 重点： 取出 qwin , kwin , 然后得到indice ， 然后从fine token 里面取数据
        # qwin : B  NumRagion C
        # kwin : B Nr C
        # 然后？ 得到indices :B Nr k
        # kfine : B N C
        # 然后？ gather : gather(indecis , kfine , vfine) -> B N k*p2 c 
        
        # com_k : shape :  (B , numregions , head , dimhead)

        # # 利用 comp_k 计算区域相似度，选取每个查询响应最高的 topk 区域
        # # sim_comp: [B, heads, N, num_regions]，其中 N = H*W, num_regions = Hc * Wc
        # _, topk_idx = sim_comp.topk(self.num_selected, dim=-1)  # [B, heads, N, num_selected]

        # # 将 fine 的 k 和 v 重构为区域块结构
        # # 先将 fine token 重构为二维形式: [B, heads, H, W, dim_head]
        # fine_k = k.view(B, self.heads, H, W, self.dim_head)
        # fine_v = v.view(B, self.heads, H, W, self.dim_head)

        # # 假设 H, W 可整除 patch_size，令 Hc = H // patch_size, Wc = W // patch_size, num_regions = Hc * Wc, patch_area = patch_size**2
        # Hc = H // self.patch_size
        # Wc = W // self.patch_size
        # num_regions = Hc * Wc
        # patch_area = self.patch_size * self.patch_size
        # N = H * W  # 例如，若 H=32, W=32，则 N=1024

        # # # 将 fine token 按照 patch_size 划分成区域，重塑为 [B, heads, Hc, patch_size, Wc, patch_size, dim_head]
        # fine_k = fine_k.view(B, self.heads, Hc, self.patch_size, Wc, self.patch_size, self.dim_head)
        # fine_v = fine_v.view(B, self.heads, Hc, self.patch_size, Wc, self.patch_size, self.dim_head)

        # # 调整维度后得到： [B, heads, num_regions, patch_area, dim_head]
        # fine_k = fine_k.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.heads, num_regions, patch_area, self.dim_head)
        # fine_v = fine_v.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.heads, num_regions, patch_area, self.dim_head)

        # # fine_k 的 shape: [B, heads, num_regions, patch_area, dim_head]
        # # fine_v 的 shape: [B, heads, num_regions, patch_area, dim_head]
        # fine_k_expanded = fine_k.unsqueeze(2).expand(B, self.heads, N, num_regions, patch_area, self.dim_head)
        # fine_v_expanded = fine_v.unsqueeze(2).expand(B, self.heads, N, num_regions, patch_area, self.dim_head)

        # # 扩展 topk_idx 到和 fine_k_expanded 匹配的形状
        # indices = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, self.heads, N, self.num_selected, patch_area, self.dim_head)
        # print(indices.shape)#torch.Size([B, heads,  N , num_selected, 1, 1])
        # print(fine_k_expanded.shape) #torch.Size([B, heads, N , num_regions, patch_area, dim_head])
        # # 沿区域维度 (dim=3) 进行 gather，结果 shape 为 [B, heads, N, num_selected, patch_area, dim_head]
        # fine_k_selected = torch.gather(fine_k_expanded, dim=3, index=indices)
        # fine_v_selected = torch.gather(fine_v_expanded, dim=3, index=indices)
        # print(fine_k_selected.shape) #torch.Size([heads, N, num_selected, 1, 1])
        # # 合并区域和 patch 内 token 数：reshape 为 [B, heads, N, num_selected * patch_area, dim_head]

        # # wrong here : RuntimeError: shape '[2, 4, 50176, 12544, 16]' is invalid for input of size 19668992
        # fine_k_selected = fine_k_selected.reshape(B, self.heads, N, self.num_selected * patch_area, self.dim_head)
        # fine_v_selected = fine_v_selected.reshape(B, self.heads, N, self.num_selected * patch_area, self.dim_head)

        # # 后续计算注意力时：
        # # q 的形状为 [B, heads, N, dim_head]
        # sim_fine = torch.matmul(q.unsqueeze(-2), fine_k_selected.transpose(-1, -2)).squeeze(-2) * scale  # [B, heads, N, num_selected*patch_area]
        # attn_fine = sim_fine.softmax(dim=-1)
        # out_fine = torch.matmul(attn_fine, fine_v_selected)  # [B, heads, N, dim_head]


        # ========= 分支 3：Sliding Window Attention =========
        # 利用 unfold 提取每个像素周围的局部邻域（二维窗口）
        # 先将 q, k, v 转换为二维形式 [B, heads, H, W, dim_head]
        # qfine_heads = qfine_heads.view(B, self.heads, H, W, self.dim_head)
        # kfine_heads = kfine_heads.view(B, self.heads, H, W, self.dim_head)
        # vfine_heads = vfine_heads.view(B, self.heads, H, W, self.dim_head)
        # 转换为 [B*heads, dim_head, H, W] 以便 unfold
        k_unfold = kfine_heads.permute(0, 1, 4, 2, 3).reshape(B * self.heads, self.dim_head, H, W)
        v_unfold = vfine_heads.permute(0, 1, 4, 2, 3).reshape(B * self.heads, self.dim_head, H, W)
        # 使用 unfold 提取局部窗口，窗口大小为 window_size，padding 保持中心对齐
        unfold = nn.Unfold(kernel_size=self.window_size, padding=self.window_size // 2)
        k_local = unfold(k_unfold)  # [B*heads, dim_head * (window_size^2), H*W]
        v_local = unfold(v_unfold)  # [B*heads, dim_head * (window_size^2), H*W]
        # 重塑为 [B, heads, dim_head, window_size^2, H, W] 后转置为 [B, heads, H, W, window_size^2, dim_head]
        k_local = k_local.view(B, self.heads, self.dim_head, self.window_size * self.window_size, H, W).permute(0,1,4,5,3,2)
        v_local = v_local.view(B, self.heads, self.dim_head, self.window_size * self.window_size, H, W).permute(0,1,4,5,3,2)
        # 对 qfine_heads 与局部 k 计算注意力
        sim_local = (qfine_heads.unsqueeze(-2) * k_local).sum(dim=-1) * self.scale  # [B, heads, H, W, window_size^2]
        attn_local = sim_local.softmax(dim=-1)
        out_local = (attn_local.unsqueeze(-1) * v_local).sum(dim=-2)  # [B, heads, H, W, dim_head]
        out_local = out_local.view(B, self.heads, H * W, self.dim_head)

        # ========= 融合三分支 =========
        # 这里简单采用平均融合，也可以设计 learnable 的融合权重
        out_total = (out_comp + out_fine + out_local) / 3.0  # [B, heads, N, dim_head]
        # 合并多头，恢复为 [B, N, total_dim]
        out_total = out_total.transpose(1, 2).reshape(B, H * W, self.total_dim)
        out_total = self.out_proj(out_total)  # [B, N, in_channels]
        # 恢复成 [B, C, H, W]
        out_total = out_total.transpose(1, 2).view(B, C, H, W)
        return out_total


# 示例测试
if __name__ == "__main__":
    model = VisionSparseAttention(
        in_channels=64,
        dim_head=16,
        heads=4,
        window_size=7,
        patch_size=7,
        num_selected=128,
    )
    x = torch.randn(2, 64, 224, 224)  # 示例输入
    y = model(x)
    print(y.shape)  # 预期输出: [2, 64, 32, 32]
