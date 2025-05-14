#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#define THREADS_PER_BLOCK 256
#include <assert.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// ----------------------------
// 1. 计算 attn 的梯度，每个线程负责 attn[b,h,r,i,j]
// ----------------------------
template <typename scalar_t>
__global__ void sparse_v_gemm_backward_attn_kernel(
    const scalar_t* __restrict__ grad_out, // [B, heads, region, w2, c]
    const scalar_t* __restrict__ vpatch,   // [B, heads, region, w2, c]
    const int* __restrict__ r_idx,         // [B, heads, region, topk]
    scalar_t* __restrict__ grad_attn,      // [B, heads, region, w2, topk]  // FIXME: [B, heads, region, topk, w2] 
    int B, int heads, int region, int w2, int topk, int c) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * heads * region * w2 * topk;
    if (index < total) {
        // 解码 index 为 (b, h, r, i, j)
        int i = index % w2;
        int tmp = index / w2;

        int j = tmp % topk;// k
        tmp = tmp / topk;

        int r = tmp % region; // r
        tmp /= region;

        int h = tmp % heads; // h
        int b = tmp / heads; // b

        scalar_t grad = 0;
        int r_idx_offset = (((b * heads + h) * region + r) * topk + j);
        int selected_region = r_idx[r_idx_offset];
        
        // 对所有通道累加
        for (int cc = 0; cc < c; cc++) {
            int grad_out_index = ((((b * heads + h) * region + r) * w2 + i) * c + cc);
            int vpatch_index = ((((b * heads + h) * region + selected_region) * w2 + i) * c + cc);
            grad += grad_out[grad_out_index] * vpatch[vpatch_index];

        }
        grad_attn[index] = grad;
    }
}

// ----------------------------
// 2. 计算 vpatch 的梯度，每个线程负责 vpatch[b,h,k,i,c]
// ----------------------------
template <typename scalar_t>
__global__ void sparse_v_gemm_backward_v_kernel(
    const scalar_t* __restrict__  grad_out, // [B, heads, region, w2, c]
    const scalar_t* __restrict__ attn,       // [B, heads, region, w2, topk] //FIXME: [B, heads, region, topk , w2]
    const int* __restrict__ r_idx,           // [B, heads, region, topk]
    scalar_t* __restrict__ grad_v,           // [B, heads, region, w2, c]
    int B, int heads, int region, int w2, int topk, int c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * heads * region * w2 * c;
    if (index < total) {
        // 解码 index 为 (b, h, k, i, cc) 其中 k 为 vpatch 的区域索引
        int cc = index % c;
        int tmp = index / c;
        int i = tmp % w2;
        tmp /= w2;
        int k_region = tmp % region;
        tmp /= region;
        int h = tmp % heads;
        int b = tmp / heads;

        scalar_t grad = 0;
        // 遍历所有可能的 (r, j)，当 r_idx[b,h,r,j] == k_region 时累加梯度
        for (int r = 0; r < region; r++) {
            for (int j = 0; j < topk; j++) {
                int r_idx_index = (((b * heads + h) * region + r) * topk + j);
                if (r_idx[r_idx_index] == k_region) {
                    int attn_index = ((((b * heads + h) * region + r) * topk + j) * w2 + i);
                    int grad_out_index = ((((b * heads + h) * region + r) * w2 + i) * c + cc);
                    grad += attn[attn_index] * grad_out[grad_out_index];
                }
            }
        }
        grad_v[index] = grad;
    }
}

// ----------------------------
// 对外接口：输入 forward 中的 grad_out, attn, vpatch, r_idx
// 返回 grad_attn 与 grad_vpatch（即 vpatch 的梯度）
// ----------------------------
std::vector<torch::Tensor> sparse_v_gemm_backward_cuda(
    torch::Tensor grad_out,   // [B, heads, region, w2, c]
    torch::Tensor attn,       // [B, heads, region, w2, topk] // FIXME:[B, heads, region,topk, w2 ]
    torch::Tensor vpatch,     // [B, heads, region, w2, c]
    torch::Tensor r_idx)      // [B, heads, region, topk]
{
    // 为 grad 输出分配空间
    auto grad_attn = torch::zeros_like(attn);
    auto grad_v = torch::zeros_like(vpatch);

    int B = attn.size(0);
    int heads = attn.size(1);
    int region = attn.size(2);
    int topk = attn.size(3);
    int w2 = attn.size(4);
    int c = vpatch.size(4);
    // 调用 attn 的反向内核
    int total_attn = B * heads * region * w2 * topk;
    int blocks_attn = (total_attn + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // printf("start attention backward attn");
    AT_DISPATCH_FLOATING_TYPES(attn.scalar_type(), "sparse_v_gemm_backward_attn", ([&] {
        sparse_v_gemm_backward_attn_kernel<scalar_t><<<blocks_attn, THREADS_PER_BLOCK>>>(
            grad_out.data_ptr<scalar_t>(),
            vpatch.data_ptr<scalar_t>(),
            r_idx.data_ptr<int>(),
            grad_attn.data_ptr<scalar_t>(),
            B, heads, region, w2, topk, c);
    }));
    // printf("nothing wrong in attention backward attn");

    // 调用 vpatch 的反向内核
    int total_v = B * heads * region * w2 * c;
    int blocks_v = (total_v + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // printf("start attention backward v");
    AT_DISPATCH_FLOATING_TYPES(vpatch.scalar_type(), "sparse_v_gemm_backward_v", ([&] {
        sparse_v_gemm_backward_v_kernel<scalar_t><<<blocks_v, THREADS_PER_BLOCK>>>(
            grad_out.data_ptr<scalar_t>(),
            attn.data_ptr<scalar_t>(),
            r_idx.data_ptr<int>(),
            grad_v.data_ptr<scalar_t>(),
            B, heads, region, w2, topk, c);
    }));
    // printf("nothing wrong in atten backward v");
    return {grad_attn, grad_v};
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("sparse_v_gemm_backward_cuda", &sparse_v_gemm_backward_cuda, "Sparse V GEMM backward (CUDA)");
// }
