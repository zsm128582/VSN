#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUDA_NUM_THREADS 256

// 简单示例：计算 qpatch 与选中 kpatch 之间的点积
// 假设 qpatch 形状为 [B, heads, region, w2, c]
// topk_index 形状为 [B, heads, region, topk]
// 输出 out 形状为 [B, heads, region, topk, w2]
// 即对于每个 (B, heads, region, w2) 上的向量，与 topk 个选中的 kpatch 向量做点积
__global__ void sparse_gemm_kernel(
    const float* __restrict__ qpatch,
    const float* __restrict__ kpatch,
    const int* __restrict__ topk_index,
    float* __restrict__ out,
    const int B,
    const int heads,
    const int region,
    const int w2,
    const int c,
    const int topk,
    const float scale)
{
    // 总线程数 = B * heads * region * w2 * topk
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * heads * region * w2 * topk;
    if(index >= total) return;
    
    int tmp = index;
    int pos = tmp % w2 ;       tmp /= w2;
    int tk  = tmp % topk;     tmp /= topk;
    int r   = tmp % region;      tmp /= region;
    int h   = tmp % heads;       tmp /= heads;
    int b   = tmp; 

    // 分解索引
    // int tmp = index;
    // int tk = tmp % topk;         tmp /= topk;
    // int pos = tmp % w2;          tmp /= w2;
    // int r   = tmp % region;      tmp /= region;
    // int h   = tmp % heads;       tmp /= heads;
    // int b   = tmp;

    // 计算 qpatch 的偏移（qpatch: [B, heads, region, w2, c]）
    int q_offset = (((b * heads + h) * region + r) * w2 + pos) * c;
    
    // 根据 topk_index 取出对应的区域索引，索引 kpatch 中的区域（kpatch: [B, heads, region, w2, c]）
    // topk_index 形状：[B, heads, region, topk]
    int topk_offset = ((b * heads + h) * region + r) * topk + tk;
    int k_r = topk_index[topk_offset];  // 这里假设 topk_index 存储的是 region 内的索引

    // kpatch 对应的偏移
    int k_offset = (((b * heads + h) * region + k_r) * w2 + pos) * c;

    // 计算点积（注意这里可以进一步优化，比如采用共享内存和并行归约）
    float sum = 0.f;
    for (int i = 0; i < c; i++) {
        float q_val = qpatch[q_offset + i];
        float k_val = kpatch[k_offset + i];
        sum += q_val * k_val;
    }
    sum *= scale;
    
    // 写入输出（out: [B, heads, region, topk, w2]）
    int out_offset = ((((b * heads + h) * region + r) * topk + tk) * w2) + pos;
    assert(out_offset == index);
    out[index] = sum;
}

torch::Tensor sparse_gemm_forward(
    torch::Tensor qpatch,
    torch::Tensor kpatch,
    torch::Tensor topk_index,
    float scale) 
{
    const auto B = qpatch.size(0);
    const auto heads = qpatch.size(1);
    const auto region = qpatch.size(2);
    const auto w2 = qpatch.size(3);
    const auto c = qpatch.size(4);
    const auto topk = topk_index.size(3);
    
    auto out = torch::zeros({B, heads, region, topk, w2}, qpatch.options());
    
    const int total_threads = B * heads * region * w2 * topk;
    const int threads = CUDA_NUM_THREADS;
    const int blocks = (total_threads + threads - 1) / threads;
    
    sparse_gemm_kernel<<<blocks, threads>>>(
        qpatch.data_ptr<float>(),
        kpatch.data_ptr<float>(),
        topk_index.data_ptr<int>(),
        out.data_ptr<float>(),
        B, heads, region, w2, c, topk, scale);
    
    return out;
}


