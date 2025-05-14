#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define THREADS_PER_BLOCK 256

// 计算 qpatch 梯度，每个线程处理 qpatch[b,h,r,i,c] 一个元素
template <typename scalar_t>
__global__ void sparse_gemm_backward_q_kernel(
    const scalar_t* __restrict__ grad_out, // [B, heads, region, topk , w2]
    const scalar_t* __restrict__ kpatch,   // [B, heads, region, w2, c]
    const int* __restrict__ r_idx,         // [B, heads, region, topk]
    scalar_t scale,
    scalar_t* __restrict__ grad_q,         // [B, heads, region, w2, c]
    int B, int heads, int region, int w2, int topk, int c) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * heads * region * w2 * c;
    if (index < total) {
        // 将 index 解码为 (b_index, h_index, r_index, w2_index, c)
        int c_index = index % c;
        int tmp = index / c;
        int w2_index = tmp % w2;
        tmp /= w2;
        int r_index = tmp % region;
        tmp /= region;
        int h_index = tmp % heads;
        int b_index = tmp / heads;

        scalar_t grad_val = 0;
        // 遍历 topk 维度
        for (int j = 0; j < topk; j++) {
            // 计算 r_idx 中对应的 kpatch 区域索引
            int r_idx_offset = (((b_index * heads + h_index) * region + r_index) * topk + j);
            int k_region = r_idx[r_idx_offset];

            // grad_out 对应位置：[b_index, h_index, r_index, w2_index, j]
            int grad_out_index = ((((b_index * heads + h_index) * region + r_index) * topk + j) * w2 + w2_index);
            scalar_t grad_out_val = grad_out[grad_out_index];

            // 对应 kpatch 的位置为：[b_index, h_index, k_region, w2_index, c_index]
            int kpatch_index = ((((b_index * heads + h_index) * region + k_region) * w2 + w2_index) * c + c_index);

            grad_val +=  grad_out_val * kpatch[kpatch_index];
        }
        
        grad_q[index] = scale * grad_val;
    }
}

// 计算 kpatch 梯度，每个线程处理 kpatch[b,h,k,i,c] 一个元素
template <typename scalar_t>
__global__ void sparse_gemm_backward_k_kernel(
    const scalar_t* __restrict__ grad_out, // [B, heads, region, topk, w2] 
    const scalar_t* __restrict__ qpatch,     // [B, heads, region, w2, c]
    const int* __restrict__ r_idx,           // [B, heads, region, topk]
    scalar_t scale,
    scalar_t* __restrict__ grad_k,           // [B, heads, region, w2, c]
    int B, int heads, int region, int w2, int topk, int c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * heads * region * w2 * c;
    if (index < total) {
        
        // 将 index 解码为 (b, h, k, i, c)，其中 k 为 kpatch 的区域索引
        int c_index = index % c;
        int tmp = index / c;
        int i = tmp % w2;
        tmp /= w2;
        int k_region = tmp % region;  // 当前 kpatch 对应的区域
        tmp /= region;
        int h = tmp % heads;
        int b = tmp / heads;

        scalar_t grad_val = 0;
        // 遍历所有 qpatch 的区域 r 和 topk 的 j
        for (int r = 0; r < region; r++) {
            for (int j = 0; j < topk; j++) {
                int r_idx_index = (((b * heads + h) * region + r) * topk + j);
                // 若该位置选中的 kpatch 区域等于当前 k_region
                if (r_idx[r_idx_index] == k_region) {
                    int grad_out_index = ((((b * heads + h) * region + r) * topk + j) * w2 + i);
                    int qpatch_index = ((((b * heads + h) * region + r) * w2 + i) * c + c_index);
                    grad_val += scale * grad_out[grad_out_index] * qpatch[qpatch_index];
                }
            }
        }

        grad_k[index] = grad_val;
    }
}

// 对外暴露的 CUDA 接口，接收 forward 过程中的 qpatch、kpatch、r_idx 及 grad_out，并返回 qpatch 与 kpatch 的梯度
std::vector<torch::Tensor> sparse_gemm_backward_cuda(
    torch::Tensor grad_out, // [B, heads, region, topk, w2]
    torch::Tensor qpatch, // [B, heads, region, w2, c]
    torch::Tensor kpatch, // [B, heads, region, w2, c]
    torch::Tensor r_idx, // [B, heads, region, topk]
    float scale) 
{

    auto grad_q = torch::zeros_like(qpatch);
    auto grad_k = torch::zeros_like(kpatch);

    int B = qpatch.size(0);
    int heads = qpatch.size(1);
    int region = qpatch.size(2);
    int w2 = qpatch.size(3);
    int c = qpatch.size(4);
    int topk = r_idx.size(3);

    int total_q = B * heads * region * w2 * c;
    int blocks_q = (total_q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // printf("start gather backward q");
    
    AT_DISPATCH_FLOATING_TYPES(qpatch.scalar_type(), "sparse_gemm_backward_q", ([&] {
        sparse_gemm_backward_q_kernel<scalar_t><<<blocks_q, THREADS_PER_BLOCK>>>(
            grad_out.data_ptr<scalar_t>(),
            kpatch.data_ptr<scalar_t>(),
            r_idx.data_ptr<int>(),
            scale,
            grad_q.data_ptr<scalar_t>(),
            B, heads, region, w2, topk, c);
    }));
    // printf("nothing wrong in gather backward q");

    // printf("start gather backward v");
    int total_k = B * heads * region * w2 * c;
    int blocks_k = (total_k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES(qpatch.scalar_type(), "sparse_gemm_backward_k", ([&] {
        sparse_gemm_backward_k_kernel<scalar_t><<<blocks_k, THREADS_PER_BLOCK>>>(
            grad_out.data_ptr<scalar_t>(),
            qpatch.data_ptr<scalar_t>(),
            r_idx.data_ptr<int>(),
            scale,
            grad_k.data_ptr<scalar_t>(),
            B, heads, region, w2, topk, c);
    }));
    // printf("nothing wrong in gather backward v");

    return {grad_q, grad_k};
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("sparse_gemm_backward_cuda", &sparse_gemm_backward_cuda, "Sparse GEMM backward (CUDA)");
// }
