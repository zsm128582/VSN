#include <cuda_runtime.h>
#include <cuda_fp16.h> // 可选支持半精度
#include <assert.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
using std::min;
// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

template <typename T>
__global__ void sparse_attention_weighting_kernel(
    const T* __restrict__ atten,      // [B, H, R, W2, K, W2]
    const T* __restrict__ value,      // [B, H, R, W2, C]
    const int64_t* __restrict__ idx,      // [B, H, R, K]
    T* __restrict__ out,              // [B, H, R, W2, C]
    const int B, const int H, const int R, 
    const int W2, const int K, const int C)
{
    // 每个 block 负责一个 (b, h, r)
    int block_id = blockIdx.x;
    int b = block_id / (H * R);
    int hr = block_id % (H * R);
    int h = hr / R;
    int r = hr % R;
    
    // 输出 tile 的总大小为 W2 * C
    int tile_size = W2 * C;
    // 线程在 block 内的线性索引
    // TODO: stride 是否有误？
    int stride = blockDim.x * blockDim.y;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每个线程采用循环分块的方式，处理自己对应的多个 tile 元素
    // 每个元素按 index 对应到 (i, c)，其中 i 属于 [0, W2)，c 属于 [0,C)
    for (int index = tid; index < tile_size; index += stride) {
        int i = index / C;  // query patch 内的像素索引（对应输出 tensor 第 3 维，范围 0~W2-1）
        int c = index % C;  // 通道索引（对应输出 tensor 最内层维度 0~C-1）
        T sum = 0;

        // 遍历当前 query patch 关联的所有 K 个 key/value 区域
        for (int j = 0; j < K; j++) {
            // 取得 idx 得到的区域下标（形状 [B, H, R, K]）
            int idx_offset = (((b * H + h) * R + r) * K) + j;
            int region_index = idx[idx_offset];

            // 利用动态共享内存载入对应 value tile，即 value[b, h, region_index, : , : ]，形状为 [W2, C]
            extern __shared__ double2 shared_memory[];
            T* shared_val = reinterpret_cast<T*>(shared_memory);

            for (int s = tid; s < tile_size; s += stride) {
                int l_tmp = s / C;   // 对应 value 的像素 l
                int cc_tmp = s % C;  // 对应通道 cc
                int value_offset = ((((b * H + h) * R + region_index) * W2) + l_tmp) * C + cc_tmp;
                shared_val[s] = value[value_offset];
            }
            __syncthreads();

            // 计算该 j 对应部分的贡献，对当前 query 像素 (i) 与通道 (c) 进行累加
            for (int l = 0; l < W2; l++) {
                // atten 的索引：atten[b,h,r,i,j,l]，
                // 注意 atten 的数据布局是假定为连续存储 [B, H, R, W2, K, W2]
                int atten_offset = ((((b * H + h) * R + r) * W2 + i) * K + j) * W2 + l;
                T a = atten[atten_offset];
                sum += a * shared_val[l * C + c]; // 共享内存中的 value tile，索引 [l, c]
            }
            __syncthreads();
        }
        // 写回输出 out[b,h,r,i,c]
        int out_offset = ((((b * H + h) * R + r) * W2 + i) * C) + c;
        out[out_offset] = sum;
    }
}

torch::Tensor weighting_forward(
    torch::Tensor attn,  // [B, H, R, W2, K , W2 ]
    torch::Tensor value,    // [B, H, R, W2, C]
    torch::Tensor idx,    // [B, H, R, K]
    float scale
)
{
    const int B = value.size(0);
    const int heads = value.size(1);
    const int region = value.size(2);
    const int W2 = value.size(3);
    const int C = value.size(4);
    const int K = idx.size(3);
    auto out = torch::zeros({B, heads, region, W2, C}, value.options());

    // 设置网格和块大小
    int gredDim = B*heads*region;
    dim3 threads_per_block(16,16); 
    

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(), "weighting_forward_cuda", ([&] {
        int shared_mem_size = (W2 * C) * sizeof(scalar_t);
        sparse_attention_weighting_kernel<scalar_t><<<gredDim, threads_per_block, shared_mem_size>>>(
            attn.data_ptr<scalar_t>(), 
            value.data_ptr<scalar_t>(), 
            idx.data_ptr<int64_t>(), 
            out.data_ptr<scalar_t>(), 
            B , heads , region , W2 , K , C
        );
    }));

    // cudaDeviceSynchronize();
    return out;
}
