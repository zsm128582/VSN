import torch
import custom_gather

class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        # 保存 forward 所需的变量（这里只保存 indices 以及 input 的 shape）
        ctx.save_for_backward(indices)
        ctx.input_shape = input.shape
        output = custom_gather.gather_cuda(input, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # 调用自定义 backward kernel，需要传入原始 input 的 shape
        grad_input = custom_gather.gather_cuda_backward(grad_output, indices, list(ctx.input_shape))
        # indices 不需要计算梯度，返回 None
        return grad_input, None

# 使用时直接调用：
def custom_gather_op(input, indices):
    return GatherFunction.apply(input, indices)

# 测试示例
if __name__ == '__main__':
    B, heads, region, w2, c = 2, 4, 8, 16, 32
    topk = 3
    input_tensor = torch.randn(B, heads, region, w2, c, device='cuda', requires_grad=True)
    indices = torch.randint(0, region, (B, heads, region, topk), device='cuda', dtype=torch.int64)

    output = custom_gather_op(input_tensor, indices)
    loss = output.sum()
    loss.backward()
    print("Backward computed successfully!")
