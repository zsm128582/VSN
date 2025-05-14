import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from typing import Tuple
from timm.models.layers import DropPath, trunc_normal_
# from localAttentionFunction import LocalAttention
from swattentionFunction import sw_qk_cuda , sw_av_cuda
from NewSparse import gatherAttention , AttentionWeighting
# from sparseAttentionFunction import gatherAttention , AttentionWeighting
# from cudaGatherFunction import GatherFunction
# import sparse_attention_cuda

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
        # FIXME: gether consume much memory , how to write a cuda version?
        B , heads , region , w2 , c =kv.size()
        # b h r k
        topk = r_idx.size(-1)
        # b h r k w2 c
        topk_kv = torch.gather(
            kv.view(B,heads,region,1,w2,c).expand(-1,-1,-1,region,-1,-1),
            dim=2,
            index=r_idx.view(B,heads,region,topk,1,1).expand(-1,-1,-1,-1,w2,c)
            )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(B , heads , region , topk , 1 ,1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        else: #'none'
            topk_kv = topk_kv # do nothing
        return topk_kv

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

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
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1) #(b h w c)
        return x

class SparseNet(nn.Module):
    def __init__(self,in_chans = 3 , num_classes = 1000 ,
                  embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                  patch_sizes=[7, 3, 3, 3],
                  window_sizes=[7, 3, 3, 3],
                  num_select=[4, 4, 4, 4],
                  num_heads=[3, 6, 12, 24],mlp_ratios=[3, 3, 3, 3],drop_path_rate=0.1,projection = 1024,isSparse=[True , True , True , False]):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        # self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
            norm_layer=nn.LayerNorm )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for ilayer in range(self.num_layers):
            layer = SpaseLayer(
                embed_dim=embed_dims[ilayer],  
                out_dim=embed_dims[ilayer+1]if (ilayer < self.num_layers - 1) else None,
                depth=depths[ilayer],
                num_heads=num_heads[ilayer],
                window_size=window_sizes[ilayer],
                patch_size=patch_sizes[ilayer], 
                num_select=num_select[ilayer],
                ffn_dim=embed_dims[ilayer] * mlp_ratios[ilayer],
                drop_path=dpr[sum(depths[:ilayer]):sum(depths[:ilayer + 1])],
                isSparse=isSparse[ilayer],
                downsample=PatchMerging if ilayer < self.num_layers - 1 else None,

            )
            self.layers.append(layer)

        self.proj = nn.Linear(self.num_features, projection)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x) #(b h w c)
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3) #(b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass
class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: bchw
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x

class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc
        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        # decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        # self.register_buffer('decay', decay)
        
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  #(n H*W H*W)
        return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None]  #(n l l)
        return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # if activate_recurrent:
        #     sin = torch.sin(self.angle * (slen[0]*slen[1] - 1))
        #     cos = torch.cos(self.angle * (slen[0]*slen[1] - 1))
        #     retention_rel_pos = ((sin, cos), self.decay.exp())

        # else:
        index = torch.arange(slen[0]*slen[1]).to(self.angle.device)
        sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
        cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)
        retention_rel_pos = (sin , cos)
            # mask_h = self.generate_1d_decay(slen[0])
            # mask_w = self.generate_1d_decay(slen[1])

            # retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        return retention_rel_pos
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x

class SpaseLayer(nn.Module):   
    def __init__(self , embed_dim , out_dim ,depth, num_heads ,window_size , patch_size , num_select, ffn_dim , drop_path ,isSparse, downsample:PatchMerging = None  ):
        super().__init__()
        self.downsample = downsample
        self.embed_dim = embed_dim
        self.depth = depth
        self.Relpos = RetNetRelPos2d(embed_dim,num_heads,2,6)

        self.blocks = nn.ModuleList([
            SparseAttentionBlock(embed_dim,num_heads,window_size,patch_size,ffn_dim , num_select,drop_path[i], isSparse )
            for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

    def forward(self, x):
        B, H, W , C= x.shape
        pos = self.Relpos((H,W))
        for blk in self.blocks:
            x = blk(x , pos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SparseAttentionBlock(nn.Module):
    def __init__(self, embed_dim , num_heads , window_size , patch_size,ffn_dim , num_selected , drop_path = 0. , isSparse = False):
        super().__init__()
        if(isSparse):
            self.attention = VisionSparseAttention(
                dim=embed_dim,
                heads=num_heads,
                window_size=window_size,
                patch_size=patch_size,
                num_selected=num_selected,
            )
        else:
            self.attention = NormalAttention(embed_dim,num_heads)
        self.embed_dim = embed_dim
        self.attentionLayerNorm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        # print(drop_path)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

    def forward(self, x , rel_pos):
        x = x+ self.pos(x)
        x = x + self.drop_path(self.attention(self.attentionLayerNorm(x) , rel_pos))
        x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x

class NormalAttention(nn.Module):
    def __init__(self, dim , heads ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Conv2d(in_channels=dim , out_channels= 3*self.dim, kernel_size=1)

    def forward(self , x , rel_pos):
        sin , cos  = rel_pos
        x = x.permute(0,3,1,2)
        B, C, H, W = x.shape

        qkv = self.to_qkv(x)  # [B,3*dim , H , W]

        qfine , kfine , vfine = qkv.chunk(3,dim= 1) #dim = b c h w
    
        #分头：
        qfine_heads = rearrange(qfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        kfine_heads = rearrange(kfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        vfine_heads = rearrange(vfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        
        qfine_heads = theta_shift(qfine_heads, sin, cos)
        kfine_heads = theta_shift(kfine_heads, sin, cos)

        qfine_heads = qfine_heads.flatten(2,3).contiguous()
        kfine_heads = kfine_heads.flatten(2,3).contiguous()
        vfine_heads = vfine_heads.flatten(2,3).contiguous()

        atten = torch.matmul(qfine_heads.view(B , self.heads , H*W , self.dim_head), kfine_heads.transpose(-1, -2)) * self.scale  # [B, heads, N, num_regions]
        logist = atten.softmax(dim=-1)
        out = torch.matmul(logist, kfine_heads)  # [B, heads, n, self.dim_head]

        out = out.transpose(1, 2).reshape(B, H * W, self.dim)
        # out_total = self.out_proj(out_total)  # [B, N, in_channels]
        # 恢复成 [B, C, H, W]
        out = out.reshape(B,H,W,-1).contiguous()
        return out



class VisionSparseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int,   # sliding window 的边长（建议为奇数，便于中心对称）
        patch_size: int,    # token compression 时的 patch 大小（假设 H,W 能被整除）
        num_selected: int,  # fine selection 分支中每个查询选择的候选区域数量
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // heads
        self.heads = heads
        self.total_dim = dim
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_selected = num_selected

        self.scale = self.dim_head ** -0.5

        # 对输入 token（展平后的像素）进行 qkv 线性映射
        # self.to_qkv = nn.Linear(in_channels, self.total_dim * 3, bias=False)
        self.to_qkv = nn.Conv2d(in_channels=dim , out_channels= 3*self.total_dim, kernel_size=1)
        # token compression 分支：利用卷积将图像分块压缩
        # self.compressConV = nn.Conv2d(in_channels=3*self.total_dim , out_channels= 3*self.total_dim ,kernel_size=patch_size , stride=patch_size )
        self.router = TopkRouting(qk_dim=self.total_dim,
                                  qk_scale=self.scale,
                                  topk=self.num_selected,
                                  diff_routing=False,
                                  param_routing=False)
        
        self.kv_gather = KVGather(mul_weight="soft")

        # self.localAttenion = LocalAttention(window_size , window_size)
        # # 输出投影
        # self.out_proj = nn.Linear(self.total_dim, in_channels)

    def forward(self, x , rel_pos):
        """
        x: [B, H, W, C]
        输出: [B, H, W, C]
        """
        # rotary embedding without mask for now
        sin , cos  = rel_pos
        x = x.permute(0,3,1,2)
        B, C, H, W = x.shape

        # # 将图像展平为 token 序列 [B, H*W, C]
        # tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]，其中 N = H*W
        # 输入 ： b c h w 1 ,64 112 112
        # 计算 q, k, v 对应高分辨率的 fine token
        qkv = self.to_qkv(x)  # [B,3*dim , H , W]

        qfine , kfine , vfine = qkv.chunk(3,dim= 1) #dim = b c h w
    
        #分头：
        qfine_heads = rearrange(qfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        kfine_heads = rearrange(kfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        vfine_heads = rearrange(vfine,"b (heads c) h w -> b heads h w c" , heads = self.heads)
        
        qfine_heads = theta_shift(qfine_heads, sin, cos)
        kfine_heads = theta_shift(kfine_heads, sin, cos)





        #TODO: 先分头会不会再gather会不会内存消耗更高？

        # 分patch 维度：B h r k w2 c
        qpatch = rearrange(qfine_heads,'b heads (j h) (i w) c  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads).contiguous()
        kpatch = rearrange(kfine_heads,'b heads (j h) (i w) c  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads).contiguous()
        vpatch = rearrange(vfine_heads,'b heads (j h) (i w) c  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads).contiguous()

        comQ = qpatch.mean(dim = -2)
        comK = kpatch.mean(dim = -2)
        comV = vpatch.mean(dim = -2)

        # 添加块级位置编码：


        # 提取patch级别的位置编码
        sin_patch = sin.reshape(H , W , -1)
        sin_patch = rearrange(sin,'(j h) (i w) d1 ->  j i  h w d1',h=self.patch_size,w=self.patch_size)
        cospatch = cos.reshape(H , W , -1)
        cospatch = rearrange(cos,'(j h) (i w) d1 ->  j i  h w d1',h=self.patch_size,w=self.patch_size)

        if(self.patch_size %2 == 1):
            center = self.patch_size // 2
        else:
            center = self.patch_size // 2 - 1
            
        sin_patch = sin_patch[:,:,center,center,:]
        cospatch = cospatch[:,:,center,center,:]


        comQ = comQ.view(B , self.heads , H // self.patch_size , W// self.patch_size, self.dim_head)
        comK = comK.view(B , self.heads , H // self.patch_size , W// self.patch_size, self.dim_head)

        comQ = theta_shift(comQ, sin_patch, cospatch)
        comK = theta_shift(comK, sin_patch, cospatch)

        comQ = comQ.view(B , self.heads , -1 , self.dim_head).contiguous()
        comK = comK.view(B , self.heads , -1 , self.dim_head).contiguous()

        Bc, _,num_regions,_= comQ.shape
        


        # 这个前置需要什么？需要kvin vvin 以及 qfine 
        # ========= 分支 1：Token Compression =========
        # 利用卷积将图像压缩成较低分辨率的区域表示
        # comp = self.compress_conv(x)  # [B, C, Hc, Wc]，其中 Hc = H/patch_size, Wc = W/patch_size
        # 位置编码：对于第一分支：qfine 和 com之间的位置编码不一样怎么办？ 所以需要对位置编码进行归一化

        sim_comp = torch.matmul(qfine_heads.view(B , self.heads , H*W , self.dim_head), comK.transpose(-1, -2)) * self.scale  # [B, heads, N, num_regions]
        attn_comp = sim_comp.softmax(dim=-1)
        out_comp = torch.matmul(attn_comp, comV)  # [B, heads, n, self.dim_head]


        # ========= 分支 2：Fine Token Selection =========
        # shape为：B p^2 k ( P2 就是块的数量)
        r_weight, r_idx = self.router(comQ, comK) # both are (n, p^2, topk) tensors


        # gatherK = GatherFunction.apply(kpatch.float() , r_idx.contiguous())
        # gatherV = GatherFunction.apply(vpatch.float() , r_idx.contiguous())
        # # b h r k w2 c
        gatherK = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kpatch) 
        gatherV = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=vpatch)
        
        # r: regions c: self.dim_heads w2: patchsize^2 k : topk 
        # 已经做好转置了
        # 这个维度还得再确认一下，为什么region要提取出来
        
        # gatherK = rearrange(gatherK,'b r k w2 (heads c) -> (b r) heads c (k w2)',heads=self.heads)
        # gatherV = rearrange(gatherV,'b r k w2 (heads c) -> (b r) heads c (k w2)',heads=self.heads) 
        # br h kw2 c 
        gatherK = rearrange(gatherK,'b h r k w2 c -> (b r) h c (k w2)')
        gatherV = rearrange(gatherV,'b h r k w2 c -> (b r) h (k w2) c')  
        # br h w2 c
        qpatch = rearrange(qpatch,'b heads r w2 c  -> (b r) heads w2 c')

        attn_fine = ((qpatch * self.scale) @ gatherK).softmax(-1)
        # br h w2 c
        out_fine = attn_fine @ gatherV
        #TODO: 这里的维度还需要确认一下
        out_fine = rearrange(out_fine,'(b r) h w2 c -> b h (r w2) c',r=num_regions)


        # # ========= 分支 2：Fine Token Selection 尝试写一个cuda算子解决 =========
        # # 这里需要 qk 均为： [B, heads, region, w2, c]

        # #得到关联patch
        # # print("x.shape",x.shape)
        # r_weight, r_idx = self.router(comQ, comK) 
        # # print("r index shape:" , r_idx.shape)
        # # print("qpatch shape:",qpatch.shape)
        # # print("kpatch shape:",kpatch.shape)
        # # 计算精细q k attention
        # # b h r w2 c
        # attn = gatherAttention.apply(qpatch.float() * self.scale , kpatch.float() , r_idx , 1).softmax(dim = -1)
        # # print("atten shape:",attn.shape)
        # # 得到最终输出
        # # b heads region w2 c
        # out_fine = AttentionWeighting.apply(attn , vpatch.float(), r_idx , 1)
        # out_fine = out_fine.flatten(2,3)
        # # out_fine = rearrange(out_fine,'b heads region w2 c -> b h (r w2) c',r=num_regions)


        # # ========= 分支 3：Sliding Window Attention =========
        # # 利用 unfold 提取每个像素周围的局部邻域（二维窗口）
        # # 转换为 [B*heads, self.dim_head, H, W] 以便 unfold

        # k_unfold = kfine_heads.permute(0, 1, 4, 2, 3).reshape(B * self.heads, self.dim_head, H, W)
        # v_unfold = vfine_heads.permute(0, 1, 4, 2, 3).reshape(B * self.heads, self.dim_head, H, W)
        # # 使用 unfold 提取局部窗口，窗口大小为 window_size，padding 保持中心对齐
        # unfold = nn.Unfold(kernel_size=self.window_size, padding=self.window_size // 2)
        # k_local = unfold(k_unfold)  # [B*heads, self.dim_head * (window_size^2), H*W]
        # v_local = unfold(v_unfold)  # [B*heads, self.dim_head * (window_size^2), H*W]
        # # 重塑为 [B, heads, self.dim_head, window_size^2, H, W] 后转置为 [B, heads, H, W, window_size^2, self.dim_head]
        # k_local = k_local.view(B, self.heads, self.dim_head, self.window_size * self.window_size, H, W).permute(0,1,4,5,3,2)
        # v_local = v_local.view(B, self.heads, self.dim_head, self.window_size * self.window_size, H, W).permute(0,1,4,5,3,2)
        # # 对 qfine_heads 与局部 k 计算注意力
        # sim_local = (qfine_heads.unsqueeze(-2) * k_local).sum(dim=-1) * self.scale  # [B, heads, H, W, window_size^2]
        # attn_local = sim_local.softmax(dim=-1)
        # out_local = (attn_local.unsqueeze(-1) * v_local).sum(dim=-2)  # [B, heads, H, W, self.dim_head]
        # out_local = out_local.view(B, self.heads, H * W, self.dim_head)


        # # ========= 分支 3：# ## 尝试使用local attention 的方式完成 =========
        
        # qfine_heads = qfine_heads.permute(0, 1, 4, 2, 3).reshape(B * self.heads, self.dim_head, H, W).contiguous().float()
        # kfine_heads = kfine_heads.permute(0, 1, 4, 2, 3).reshape(B * self.heads, self.dim_head, H, W).contiguous().float()
        # vfine_heads = vfine_heads.permute(0, 1, 4, 2, 3).reshape(B * self.heads, self.dim_head, H, W).contiguous().float()
        # out_local = self.localAttenion(qfine_heads, kfine_heads , vfine_heads)
        # out_local = rearrange(out_local , "(b heads ) c h w -> b heads (h w) c " , heads = self.heads)
        # # print(out_local.shape)
        # 显存明显降低，但是计算速度仍然非常慢

        # # ========= 分支 3：# ## ##尝试使用TransNet的方法完成 =========
        
        # B , h , N  , C
        qfine_heads = qfine_heads.reshape(B, self.heads , -1 , self.dim_head).contiguous()
        kfine_heads = kfine_heads.reshape(B, self.heads , -1 , self.dim_head).contiguous()
        vfine_heads = vfine_heads.reshape(B, self.heads , -1 , self.dim_head).contiguous()
        attn_local = sw_qk_cuda.apply(qfine_heads,kfine_heads , H , W , self.window_size)
        out_local = sw_av_cuda.apply(attn_local.type_as(vfine_heads) , vfine_heads , H ,W , self.window_size)
        # 我靠不愧是Daishi！大师受我一拜

        # ========= 融合三分支 =========
        # 这里简单采用平均融合，也可以设计 learnable 的融合权重
        out_total = (out_comp  + out_fine + out_local )/ 3# [B, heads, N, self.dim_head]
        # 合并多头，恢复为 [B, N, total_dim]
        out_total = out_total.transpose(1, 2).reshape(B, H * W, self.total_dim)
        # out_total = self.out_proj(out_total)  # [B, N, in_channels]
        # 恢复成 [B, C, H, W]
        out_total = out_total.reshape(B,H,W,-1)
        return out_total


from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
@register_model
def VSN(arg):
    model = SparseNet(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        patch_sizes=[8,4,2 , 1],
        window_sizes=[7,5,3 , 1],
        num_select=[12,12,12,12],
        num_heads=[4,4,8,8],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.15,
        projection=1024,
        isSparse=[True,True,True,False]
    )
    model.default_cfg = _cfg()
    return model

# 示例测试
if __name__ == "__main__":
    model = VisionSparseAttention(
        dim=64,
        heads=4,
        window_size=15,
        patch_size=16,
        num_selected=12,
    ).cuda()

    model = VSN().cuda()
    x = torch.randn(1,3,224,224 ).cuda()  # 示例输入
    y = model(x)
    print(y.shape)  # 预期输出: [2, 64, 32, 32]
    # # from thop import profile
    # x = torch.randn(2, 64, 112, 112).cuda()  # 示例输入
    # relpos = RetNetRelPos2d(64,4,2,6).cuda()
    # pos = relpos((112,112))

    # from torch_flops import TorchFLOPsByFX
    # flops_counter = TorchFLOPsByFX(model)
    # flops_counter.propagate(x)

    # flops_counter.print_result_table()
    # flops_1 = flops_counter.print_total_flops(show=False)
    # print(f"torch_flops: {flops_1} FLOPs")
    # print("=" * 80)

    # flops, params = profile(model, inputs=(x,))
    # print(f"FLOPs: {flops}, Params: {params}")
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)

    # start_event.record()
    # y = model(x , pos)
    # end_event.record()
    print(y.shape)  # 预期输出: [2, 64, 32, 32]
    # torch.cuda.synchronize()
    # print(start_event.elapsed_time(end_event))  # 计算耗时

    # from ptflops import get_model_complexity_info
    # macs , params = get_model_complexity_info(model,(64,112,112) ,as_strings=True , print_per_layer_stat= True ,verbose=True)
    # print('model flops:', macs)