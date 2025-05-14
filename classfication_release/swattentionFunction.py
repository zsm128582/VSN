import torch
import swattention


CUDA_NUM_THREADS = 128


class sw_qk_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, height, width, kernel_size):
        attn_weight = swattention.qk_forward(query, key, height, width, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(query, key)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size

        return attn_weight

    @staticmethod
    def backward(ctx, d_attn_weight):
        query, key = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size

        d_query, d_key = swattention.qk_backward(d_attn_weight.contiguous(), query, key, height, width,
                                                            kernel_size, CUDA_NUM_THREADS)

        return d_query, d_key, None, None, None


class sw_av_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, height, width, kernel_size):
        output = swattention.av_forward(attn_weight, value, height, width, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(attn_weight, value)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size

        return output

    @staticmethod
    def backward(ctx, d_output):
        attn_weight, value = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size

        d_attn_weight, d_value = swattention.av_backward(d_output.contiguous(), attn_weight, value, height, width,
                                                         kernel_size, CUDA_NUM_THREADS)

        return d_attn_weight, d_value, None, None, None
