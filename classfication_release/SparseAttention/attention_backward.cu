#include <cuda_runtime.h>
#include <cstdio>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>
// 反向传播：计算 query 的梯度
// d_atten: 上游梯度, shape [B, H, R, W2, K, W2]
// key:     前向的 key, shape [B, H, R, W2, C]
// idx:     前向中用于选择 key 的 index, shape [B, H, R, K]
// grad_query: 输出梯度, shape [B, H, R, W2, C]
//
// 每个 block 处理一个 region (b, h, r)，线程二维分布遍历 query 内像素 i 和通道 c。
template <typename scalar_t>
__global__ void sparse_attention_backward_query_kernel(
    const scalar_t* __restrict__ d_atten, // [B, H, R, W2, K, W2]
    const scalar_t* __restrict__ key,     // [B, H, R, W2, C]
    const int64_t*   __restrict__ idx,      // [B, H, R, K]
    scalar_t*       __restrict__ grad_query, // [B, H, R, W2, C]
    int B, int H, int R, int W2, int C, int K)
{
    // 每个 block 处理一个 region (b, h, r)
    int r = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;


    for (int i = threadIdx.x; i < W2; i += blockDim.x) {
        for (int c = threadIdx.y; c < C; c += blockDim.y) {
            float grad_val = 0.0f;
            // 对每个关联的 key 区域（j）以及 key 内像素 l 累加贡献
            for (int j = 0; j < K; j++) {
                // 读取 idx 得到 key 中的 region 下标
                int idx_offset = ((b * H + h) * R + r) * K + j;
                int64_t key_region = idx[idx_offset];
                for (int l = 0; l < W2; l++) {
                    // d_atten[b, h, r, i, j, l]
                    int atten_offset = ((((b * H + h) * R + r) * W2 + i) * K + j) * W2 + l;
                    // key[b, h, key_region, l, c]
                    int key_offset = (((b * H + h) * R + key_region) * W2 + l) * C + c;
                    grad_val += d_atten[atten_offset] * key[key_offset];
                }
            }
            int query_offset = (((b * H + h) * R + r) * W2 + i) * C + c;
            grad_query[query_offset] = grad_val;
        }
    }
}

// 反向传播：计算 key 的梯度
// d_atten: 上游梯度, shape [B, H, R, W2, K, W2]
// query:   前向的 query, shape [B, H, R, W2, C]
// idx:     前向中用于选择 key 的 index, shape [B, H, R, K]
// grad_key: 输出梯度, shape [B, H, R, W2, C]
// 注意：由于多个 region 可能会更新同一个 key 元素，因此需要使用 atomicAdd
template <typename scalar_t>
__global__ void sparse_attention_backward_key_kernel(
    const scalar_t* __restrict__ d_atten, // [B, H, R, W2, K, W2]
    const scalar_t* __restrict__ query,   // [B, H, R, W2, C]
    const int64_t*   __restrict__ idx,      // [B, H, R, K]
    scalar_t*       __restrict__ grad_key, // [B, H, R, W2, C]
    int B, int H, int R, int W2, int C, int K)
{
    // 每个 block 处理一个 region (b, h, r)
    int r = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int numel = B*H*R*W2*C;
    // 遍历 query 内像素 i 和 key 内像素 l（注意：这里 l 表示的是 d_atten 内用于计算 key 对应位置的维度）
    for (int i = threadIdx.x; i < W2; i += blockDim.x) {
        for (int l = threadIdx.y; l < W2; l += blockDim.y) {
            // 对每个关联区域 j
            for (int j = 0; j < K; j++) {
                // 得到 key 中对应的 region 下标
                int idx_offset = ((b * H + h) * R + r) * K + j;
                int64_t key_region = idx[idx_offset];
                // 对所有通道 c 累加梯度贡献
                for (int c = 0; c < C; c++) {
                    // 上游梯度 d_atten[b, h, r, i, j, l]
                    int atten_offset = ((((b * H + h) * R + r) * W2 + i) * K + j) * W2 + l;
                    scalar_t grad_atten_val = d_atten[atten_offset];
                    // query[b, h, r, i, c]
                    int query_offset = (((b * H + h) * R + r) * W2 + i) * C + c;
                    scalar_t query_val = query[query_offset];
                    scalar_t grad_contrib = grad_atten_val * query_val;
                    // 累加到 grad_key[b, h, key_region, l, c]
                    int key_offset = (((b * H + h) * R + key_region) * W2 + l) * C + c;
                    // atomicAdd(&grad_key[key_offset], grad_contrib);
                    at::native::fastAtomicAdd(grad_key, key_offset , numel , static_cast<scalar_t>(grad_contrib) ,true);
                }
            }
        }
    }
}


std::vector<torch::Tensor> sparse_attention_backward(
    torch::Tensor d_atten,   // [B, H, R, W2, K, W2]
    torch::Tensor query,     // [B, H, R, W2, C]
    torch::Tensor key,       // [B, H, R, W2, C]
    torch::Tensor idx,       // [B, H, R, K]
    float scale             // 缩放因子
)       
{
    int B = d_atten.size(0);
    int H = d_atten.size(1);
    int R = d_atten.size(2);
    int W2 = d_atten.size(3);
    int K = d_atten.size(4);
    int C = query.size(4);

    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);

    // dim3 block(W2, C); 
    // dim3 grid(R, H, B); 
    // 对于 query 反向传播核：假设 W2 和 C 较大时采用分块处理
    dim3 block_query(16, 16);  // 16x16 = 256 threads per block
    dim3 grid_query(R, H, B);  // 每个 block 处理一个 region
    // 对于 key 反向传播核：假设 W2 较大时采用分块处理
    dim3 block_key(16, 16);    // 16x16 = 256 threads per block
    dim3 grid_key(R, H, B);    // 每个 block 处理一个 region
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(d_atten.scalar_type(), "sparse_attention_backward_kernel", ([&] {
        sparse_attention_backward_query_kernel<scalar_t><<<grid_query, block_query>>>(
            d_atten.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            idx.data_ptr<int64_t>(),
            grad_query.data_ptr<scalar_t>(),
            B, H, R, W2, C, K
        );
        sparse_attention_backward_key_kernel<scalar_t><<<grid_key, block_key>>>(
            d_atten.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            idx.data_ptr<int64_t>(),
            grad_key.data_ptr<scalar_t>(),
            B, H, R, W2, C, K
        );

    }));

    return {grad_query.contiguous(), grad_key.contiguous()};
}


// // 完整的反向传播函数
// void sparse_attention_backward(
//     const float* d_atten,   // 上游梯度, shape: [B, H, R, W2, K, W2]
//     const float* query,     // 前向 query,   shape: [B, H, R, W2, C]
//     const float* key,       // 前向 key,     shape: [B, H, R, W2, C]
//     const int* idx,         // idx,          shape: [B, H, R, K]
//     float* grad_query,      // 输出: grad_query, shape: [B, H, R, W2, C]
//     float* grad_key,        // 输出: grad_key,   shape: [B, H, R, W2, C]
//     int B, int H, int R, int W2, int C, int K)
// {
//     // 注意：在调用 backward_key kernel 前，需要将 grad_key 内存置 0，
//     // 假定 grad_key_total = B * H * R * W2 * C 个 float
//     cudaMemset(grad_key, 0, sizeof(float) * B * H * R * W2 * C);

//     // 设置 grid: 每个 block 处理一个 region (b, h, r)
//     dim3 grid(R, H, B);

//     // 对于 backward_query_kernel，建议采用二维 block，例如 (16,16)
//     // 内核中建议采用循环来覆盖所有 query 内像素 (i) 与通道 (c) ，例如：
//     //   for (int i = threadIdx.x; i < W2; i += blockDim.x)
//     //   for (int c = threadIdx.y; c < C; c += blockDim.y)
//     dim3 block_query(16, 16);
//     sparse_attention_backward_query_kernel<<<grid, block_query>>>(d_atten, key, idx, grad_query, B, H, R, W2, C, K);

//     // 对于 backward_key_kernel，同样采用二维 block (16,16)
//     // 内核中需通过循环覆盖 query 内像素 i 与 key 内像素 l（如果 W2 超过 16）
//     dim3 block_key(16, 16);
//     sparse_attention_backward_key_kernel<<<grid, block_key>>>(d_atten, query, idx, grad_key, B, H, R, W2, C, K);

//     // 同步，检查 kernel 是否执行正确
//     cudaDeviceSynchronize();
// }