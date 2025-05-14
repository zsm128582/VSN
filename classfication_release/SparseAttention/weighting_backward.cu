#include <cuda_runtime.h>
#include <cuda_fp16.h> // 可选支持半精度
#include <assert.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>
// 后向 kernel 1：计算 grad_atten（不涉及写冲突，可直接计算）
// 对于每个 atten 元素，其梯度：
// grad_atten[b,h,r,i,j,l] = sum_{c} grad_out[b,h,r,i,c] * value[b,h, idx[b,h,r,j], l, c]
template <typename scalar_t>
__global__ void attention_weighting_backward_grad_atten_kernel(
    const scalar_t* __restrict__ grad_out,  // [B, H, R, W2, C]
    const scalar_t* __restrict__ value,       // [B, H, R, W2, C]
    const int64_t* __restrict__ idx,           // [B, H, R, K]
    scalar_t* __restrict__ grad_atten,        // [B, H, R, W2, K, W2]
    int B, int H, int R, int W2, int K, int C)
{
    // 每个线程负责一个 grad_atten 元素
    int total = B * H * R * W2 * K * W2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    
    // 将 tid 解码为 (b, h, r, i, j, l)
    int l = tid % W2;
    int j = (tid / W2) % K;
    int i = (tid / (W2 * K)) % W2;
    int r = (tid / (W2 * K * W2)) % R;
    int h = (tid / (W2 * K * W2 * R)) % H;
    int b = tid / (W2 * K * W2 * R * H);
    
    int idx_offset = (((b * H + h) * R + r) * K) + j;
    int64_t region_index = idx[idx_offset];
    
    float sum = 0.0f;
    // 对 c 求和
    for (int c = 0; c < C; c++) {
        int grad_out_offset = ((((b * H + h) * R + r) * W2 + i) * C) + c;
        int value_offset = ((((b * H + h) * R + region_index) * W2 + l) * C) + c;
        sum += grad_out[grad_out_offset] * value[value_offset];
    }
    int out_offset = ((((b * H + h) * R + r) * W2 + i) * K + j) * W2 + l;
    grad_atten[out_offset] = sum;
}

// 后向 kernel 2：计算 grad_value（scatter 操作，同一 value 元素可能被多个线程更新，因此使用 atomicAdd）
// 每个线程处理 (b,h,r,i,j,l,c) 上的贡献：
// grad_value[b,h, idx[b,h,r,j], l, c] += atten[b,h,r,i,j,l] * grad_out[b,h,r,i,c]
template <typename scalar_t>
__global__ void attention_weighting_backward_grad_value_kernel(
    const scalar_t* __restrict__ grad_out,  // [B, H, R, W2, C]
    const scalar_t* __restrict__ atten,       // [B, H, R, W2, K, W2]
    const int64_t* __restrict__ idx,           // [B, H, R, K]
    scalar_t* __restrict__ grad_value,        // [B, H, R, W2, C]
    int B, int H, int R, int W2, int K, int C)
{
    int numel = B * H * R * W2 * C;
    // int total = B * H * R * W2 * K * W2 * C;
    int total = numel * K * W2;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    
    // 解码 tid 为 (b, h, r, i, j, l, c)
    int c = tid % C;
    int l = (tid / C) % W2;
    int j = (tid / (C * W2)) % K;
    int i = (tid / (C * W2 * K)) % W2;
    int r = (tid / (C * W2 * K * W2)) % R;
    int h = (tid / (C * W2 * K * W2 * R)) % H;
    int b = tid / (C * W2 * K * W2 * R * H);
    
    int idx_offset = (((b * H + h) * R + r) * K) + j;
    int64_t region_index = idx[idx_offset];
    
    int grad_out_offset = ((((b * H + h) * R + r) * W2 + i) * C) + c;
    scalar_t grad_out_val = grad_out[grad_out_offset];
    
    int atten_offset = ((((b * H + h) * R + r) * W2 + i) * K + j) * W2 + l;
    scalar_t a = atten[atten_offset];
    
    scalar_t grad = a * grad_out_val;
    
    // 累加到 grad_value[b, h, region_index, l, c]（注意可能存在写冲突）
    int grad_val_offset = ((((b * H + h) * R + region_index) * W2 + l) * C) + c;
    at::native::fastAtomicAdd(grad_value, grad_val_offset , numel , static_cast<scalar_t>(grad) , true );
    // atomicAdd(&grad_value[grad_val_offset], grad);
}


std::vector<torch::Tensor> sparse_weighting_backward(
    torch::Tensor d_out,   // [B, H, R, W2, C]
    torch::Tensor atten,     // [B, H, R, W2,  K , W2]
    torch::Tensor value,       // [B, H, R, W2, C]
    torch::Tensor idx,       // [B, H, R, K]
    float scale             // 缩放因子
)       
{
    int B = d_out.size(0);
    int H = d_out.size(1);
    int R = d_out.size(2);
    int W2 = d_out.size(3);
    int K = idx.size(3);
    int C = d_out.size(4);

    auto grad_attn = torch::zeros_like(atten);
    auto grad_value = torch::zeros_like(value);

    int threads = 256;


    int total_grad_atten = B * H * R * W2 * K * W2;
    int attn_blocks = (total_grad_atten + threads - 1) / threads;

    int total_grad_value = B * H * R * W2 * K * W2 * C;
    int value_blocks = (total_grad_value + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(d_out.scalar_type(), "attention_weighting_backward_grad_value_kernel", ([&] {
        attention_weighting_backward_grad_value_kernel<scalar_t><<<value_blocks, threads>>>(
            d_out.data_ptr<scalar_t>(),
            atten.data_ptr<scalar_t>(),
            idx.data_ptr<int64_t>(),
            grad_value.data_ptr<scalar_t>(),
            B, H, R, W2, K , C
        );
        attention_weighting_backward_grad_atten_kernel<scalar_t><<<attn_blocks, threads>>>(
            d_out.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            idx.data_ptr<int64_t>(),
            grad_attn.data_ptr<scalar_t>(),
            B, H, R, W2, K , C
        );
    }));

    // AT_DISPATCH_FLOATING_TYPES(d_out.scalar_type(), "attention_weighting_backward_grad_atten_kernel", ([&] {
    //     attention_weighting_backward_grad_atten_kernel<scalar_t><<<attn_blocks , threads>>>(
    //         d_out.data_ptr<scalar_t>(),
    //         value.data_ptr<scalar_t>(),
    //         idx.data_ptr<int64_t>(),
    //         grad_attn.data_ptr<scalar_t>(),
    //         B, H, R, W2, K , C
    //     );
    // }));



    // cudaDeviceSynchronize();
    return {grad_attn.contiguous(), grad_value.contiguous()};
}