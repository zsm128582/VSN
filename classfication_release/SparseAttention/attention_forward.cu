#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <torch/extension.h>
#include <cuda_fp16.h>
using std::min;

// 优化后的 CUDA 内核，使用共享内存对 key 区域数据进行复用
// 输入：
//   query: [B, H, R, W2, C]
//   key:   [B, H, R, W2, C]
//   idx:   [B, H, R, K]  每个元素给出 key 中的 region 下标
// 输出：
//   atten: [B, H, R, W2, K, W2]
// 参数：B, H, R, W2, C, K
template <typename T>
__global__ void sparse_attention_kernel(
    const T* __restrict__ query, // [B, H, R, W2, C]
    const T* __restrict__ key,   // [B, H, R, W2, C]
    const int64_t*   __restrict__ idx,    // [B, H, R, K]
    T*       __restrict__ atten,  // [B, H, R, W2, K, W2]
    const int B,const int H,const int R,const int W2,const int C,const int K ,const float scale)
{
    // 每个 block 处理一个 region: (b, h, r)
    int r = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
        // 动态共享内存，用于存放当前 tile 内所有 K 个 key 区域的数据
    // 共享内存布局：shared_key[j][l_idx][t] 线性化为：
    //   shared_key[ j * (W2 * current_tile) + l_idx * current_tile + t ]
    extern __shared__ double2 shared_memory[];
    T* shared_key = reinterpret_cast<T*>(shared_memory);
    // shared_key = (scalar_t*) shared_key;
    
    // 这里我们采用二维线程块来遍历 query 内像素 i 和 key 内像素 l
    // 每个线程将计算输出 atten[b, h, r, i, :, l]（即所有 K 个点积结果）
    // 由于 region 内的 W2 可能较大（16 或 64），采用分块扫描：
    for (int i = threadIdx.x; i < W2; i += blockDim.x) {
        for (int l = threadIdx.y; l < W2; l += blockDim.y) {
            // 每个线程为所有 j (0 <= j < K) 维护一个累加器
            T accum[12]; // 假定 K 最大为 12
            for (int j = 0; j < K; j++) {
                accum[j] = 0;
            }
            
            // 我们在通道维度上做 tile，每次处理 TILE_C 个通道
            const int TILE_C = 16;  
            // 遍历 C 维（注意可能 C 不是 TILE_C 的倍数）
            for (int tile_c = 0; tile_c < C; tile_c += TILE_C) {
                int current_tile = min(TILE_C, C - tile_c);

                

                // 每个 block 内所有线程协作将 key 数据加载到共享内存
                // 注意：对于同一 region，query 中的每个输出元素需要对应 key 中同一组 [K, W2, current_tile] 数据
                int total_elements = K * W2 * current_tile;
                // 使用线程块内二维线程联合索引构造全局线程索引
                int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
                int block_threads = blockDim.x * blockDim.y;
                // printf(" ok here");
                for (int idx_flat = thread_id; idx_flat < total_elements; idx_flat += block_threads) {
                    // 反解出 j, l_idx, t
                    int j = idx_flat / (W2 * current_tile);
                    int rem = idx_flat % (W2 * current_tile);
                    int l_idx = rem / current_tile;
                    int t = rem % current_tile;
                    
                    // 根据 idx 读取 key 中对应 region 的下标
                    int idx_offset = ((b * H + h) * R + r) * K + j;
                    int64_t key_region = idx[idx_offset];
                    // key 的全局索引： key[b, h, key_region, l_idx, (tile_c + t)]
                    int key_offset = (((b * H + h) * R + key_region) * W2 + l_idx) * C + (tile_c + t);
                    assert(idx_flat < K * W2 * TILE_C );
                    shared_key[idx_flat] = key[key_offset];
                }
                __syncthreads();
                
                // 加载 query 当前 tile 内的数值
                // query[b, h, r, i, channel] 对应 channel = tile_c ... tile_c+current_tile-1
                T q_tile[16];  // 临时存储 tile 内 query 的值
                for (int t = 0; t < current_tile; t++) {
                    int channel = tile_c + t;
                    int query_offset = (((b * H + h) * R + r) * W2 + i) * C + channel;
                    q_tile[t] = query[query_offset];
                }
                
                // 对当前 tile 内的所有通道累加点积：
                // 对于每个关联的 j (0<=j<K)，我们从共享内存中取出 key 对应值：
                //   key_val = shared_key[ j * (W2 * current_tile) + l * current_tile + t ]
                // 累加： accum[j] += q_tile[t] * key_val
                for (int t = 0; t < current_tile; t++) {
                    for (int j = 0; j < K; j++) {
                        assert(j * (W2 * current_tile) + l * current_tile + t < K * W2 * TILE_C);
                        T key_val = shared_key[j * (W2 * current_tile) + l * current_tile + t];
                        accum[j] += q_tile[t] * key_val;
                    }
                }
                __syncthreads();
            } // end for tile_c

            // 将累加结果写回输出 atten[b, h, r, i, j, l]，对于所有 j
            for (int j = 0; j < K; j++) {
                int out_offset = ((((b * H + h) * R + r) * W2 + i) * K * W2) + j * W2 + l;
                atten[out_offset] = accum[j];
            }
        }
    }
}


torch::Tensor sparse_attention(
    torch::Tensor query,  // [B, H, R, W2, C]
    torch::Tensor key,    // [B, H, R, W2, C]
    torch::Tensor idx,    // [B, H, R, K]
    float scale
)
{
    const auto B = query.size(0);
    const auto heads = query.size(1);
    const auto region = query.size(2);
    const auto W2 = query.size(3);
    const auto C = query.size(4);
    const auto K = idx.size(3);

    auto atten = torch::zeros({B, heads, region, W2, K , W2}, query.options());
    // 使用 3D grid：
    //   grid.x = R (region 数)
    //   grid.y = H (head 数)
    //   grid.z = B (batch 大小)
    dim3 grid(region, heads, B);
    dim3 block(16,16);
    int TILE_C = 16;
    
    // // 每个 block 内的线程数可根据实际情况调节
    // int threads = 256;
    
    // 调用 kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "sparse_attention_cuda", ([&] {
        int shared_mem_size = K * W2 * TILE_C * sizeof(scalar_t);
        sparse_attention_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            query.data_ptr<scalar_t>(), 
            key.data_ptr<scalar_t>(), 
            idx.data_ptr<int64_t>(), 
            atten.data_ptr<scalar_t>(), 
            B, heads, region, W2, C, K , scale
        );
    }));
    // cudaDeviceSynchronize();
    return atten;
}
