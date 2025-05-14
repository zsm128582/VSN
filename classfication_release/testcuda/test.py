import sys
import os

# 获取当前文件（sparseTest.py）的绝对路径，然后找到主目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sparseAttentionFunction import gatherAttention , AttentionWeighting
import torch
from cudaGatherFunction import GatherFunction

from einops import rearrange
from SpaseNet import TopkRouting , KVGather

def cuda_apply(qpatch , kpatch , vpatch , scale , num_selected , r_idx):
    gatherK = GatherFunction.apply(kpatch, r_idx)
    return gatherK



def pytorch_apply(qpatch , kpatch , vpatch , scale , num_selected , r_idx):
    B, heads, region, w2, c = kpatch.shape
    gatherK = torch.gather(
        kpatch.view(B,heads,region,1,w2,c).expand(-1,-1,-1,region,-1,-1),
        dim=2,
        index=r_idx.view(B,heads,region,num_selected,1,1).expand(-1,-1,-1,-1,w2,c)
        )
    return gatherK
    

def forwardTest():
    B = 1
    H = 1
    REGION = 4
    W2 = 4
    C = 1
    scale = 1
    num_selected = 2
    qpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    kpatch= torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    vpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    comQ = qpatch.mean(dim = -2).contiguous()
    comK = kpatch.mean(dim = -2).contiguous()
    comV = vpatch.mean(dim = -2).contiguous()

    router = TopkRouting(qk_dim=C,
                                qk_scale=scale,
                                topk=num_selected,
                                diff_routing=False,
                                param_routing=False)
    r_weight, r_idx = router(comQ, comK) # both are (n, p^2, topk) tensors

    # 为了方便调试，将kpatch的值设置为0到15
    kpatch =  torch.randn(16)
    for i in range(16):
        kpatch[i] = i

    kpatch = kpatch.view(1, 1, 4, 4, 1).cuda().contiguous()
    print("kpatch:")
    print(kpatch)
    print("r_idx:")
    print(r_idx)

    print("cuda Result:")
    cudaResult = cuda_apply( qpatch , kpatch , vpatch , scale , num_selected , r_idx.contiguous())
    # 为了结果方便阅读，将大小为1的维度如 b , h , c压缩掉
    cudaResult = cudaResult.squeeze(0).squeeze(0).squeeze(-1)

    print(cudaResult)

    print("pytorch Result:")
    pytorchResult = pytorch_apply( qpatch , kpatch , vpatch , scale , num_selected,r_idx.contiguous())

    pytorchResult = pytorchResult.squeeze(0).squeeze(0).squeeze(-1)
    print(pytorchResult)
    
    assert torch.allclose(cudaResult, pytorchResult, atol=1e-5), "CUDA and PyTorch results do not match!"

def checkBackward():
    # 构造测试数据
    # B, H, R, W2, C = 4, 4, 49, 64, 64
    B, H, R, W2, C = 1, 1, 49, 64, 64
    num_selected = 12
    # 使用双精度，并确保需要梯度
    input_tensor = torch.randn(B, H, R, W2, C, dtype=torch.float64, device='cuda', requires_grad=True)
    # 构造一个固定的 r_idx ，无需梯度
    r_idx = torch.randint(0, R, (B, H, R, num_selected), device='cuda', dtype=torch.int64)
    # gatherk = GatherFunction.apply(input_tensor, r_idx)
    # print(gatherk.shape)
    # from torch.autograd import gradcheck
    # # 使用 gradcheck 检查
    # test = gradcheck(GatherFunction.apply, (input_tensor, r_idx), eps=1e-6, atol=1e-4)
    # print("Gradcheck passed?", test)

if __name__ == "__main__":
    # forwardTest()
    checkBackward()
    # testGatherFunction()