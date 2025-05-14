#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUDA_NUM_THREADS 256

// 内核实现：对每个 (b, h, r, pos, c) 计算加权和
// attn: [B, heads, region, topk, w2]
// vpatch: [B, heads, region, w2, c]
// topk_index: [B, heads, region, topk]
// out: [B, heads, region, w2, c]
__global__ void sparse_v_gemm_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ vpatch,
    const int* __restrict__ topk_index,
    float* __restrict__ out,
    const int B,
    const int heads,
    const int region,
    const int w2,
    const int c,
    const int topk)
{
    // 每个线程负责计算一个输出元素 (b, h, r, pos, c)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * heads * region * w2 * c;
    if (index >= total) return;
    
    // 分解索引
    int tmp = index;
    int col = tmp % c;   // 特征维度索引
    tmp /= c;
    int pos = tmp % w2;  // 区域内细粒度位置索引
    tmp /= w2;
    int r = tmp % region;  // 区域索引（query 对应的区域）
    tmp /= region;
    int h = tmp % heads;
    tmp /= heads;
    int b = tmp;
    
    float sum = 0.f;
    // 对于每个 topk 选出的区域，做加权累加
    for (int tk = 0; tk < topk; tk++) {
        // 从 topk_index 中取出对应区域索引
        int topk_offset = ((b * heads + h) * region + r) * topk + tk;
        int v_region = topk_index[topk_offset];  // vpatch 中选出的区域索引
        
        // vpatch 的索引：我们取的是 vpatch[b, h, v_region, pos, c]
        int v_offset = (((b * heads + h) * region + v_region) * w2 + pos) * c + col;
        float v_val = vpatch[v_offset];
        
        // attn 对应索引：attn[b, h, r, tk, pos]
        int attn_offset = (((b * heads + h) * region + r) * topk + tk) * w2 + pos;
        float attn_val = attn[attn_offset];
        
        sum += attn_val * v_val;
    }
    
    // 写入输出
    int out_offset = (((b * heads + h) * region + r) * w2 + pos) * c + col;
    out[out_offset] = sum;
}

torch::Tensor sparse_v_gemm_forward(
    torch::Tensor attn,      // [B, heads, region, topk, w2]
    torch::Tensor vpatch,    // [B, heads, region, w2, c]
    torch::Tensor topk_index,// [B, heads, region, topk]
    int topk)
{
    const int B = attn.size(0);
    const int heads = attn.size(1);
    const int region = attn.size(2);
    const int w2 = attn.size(4);
    const int c = vpatch.size(4);
    
    // 创建输出张量
    auto out = torch::zeros({B, heads, region, w2, c}, attn.options());
    
    const int total = B * heads * region * w2 * c;
    const int threads = CUDA_NUM_THREADS;
    const int blocks = (total + threads - 1) / threads;
    
    sparse_v_gemm_kernel<<<blocks, threads>>>(
        attn.data_ptr<float>(),
        vpatch.data_ptr<float>(),
        topk_index.data_ptr<int>(),
        out.data_ptr<float>(),
        B, heads, region, w2, c, topk);
    
    return out;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("sparse_v_gemm_forward", &sparse_v_gemm_forward, "Sparse V GEMM forward (CUDA)");
// }
