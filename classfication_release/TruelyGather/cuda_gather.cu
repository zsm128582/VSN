#include <torch/extension.h>
#include <vector>
#include <stdio.h>
// forward CUDA kernel：根据 topk 索引从输入 tensor 中拷贝数据
template <typename scalar_t>
__global__ void gather_cuda_kernel(
    const scalar_t* __restrict__ input,   // 输入 tensor: [B, heads, region, w2, c]
    const int64_t* __restrict__ indices,  // topk 索引: [B, heads, region, topk]
    scalar_t* __restrict__ output,        // 输出 tensor: [B, heads, region, topk, w2, c]
    int B, int heads, int region, int topk, int w2, int c) 
{
    
    int total = B * heads * region * topk ;
    if (idx >= total) return;

    int ik = idx % topk;           int tmp = idx / topk;
    int ir = tmp % region;         tmp /= region;
    int ih = tmp % heads;          
    int ib = tmp / heads;

    // 获取当前 (ib, ih, ir, itopk) 对应的目标 region 索引
    int64_t src_r = indices[((ib * heads + ih) * region + ir) * topk + ik];
    // printf("%d \n", src_r);
    int out_offset = (((ib * heads + ih) * region + ir) * topk + ik) * (w2 * c);
    int in_offset = ((ib * heads + ih) * region + src_r) * (w2 * c);

    // 连续复制内存块，访存会更快一些
    for(int i = 0 ; i < w2 * c; i++) {
        output[out_offset + i] = input[in_offset + i];
        // printf("%f ", input[in_offset + i] );
        
    }


}

// backward CUDA kernel：将 grad_output 中的梯度散射回 input 的对应位置
template <typename scalar_t>
__global__ void gather_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_output, // [B, heads, region, topk, w2, c]
    const int64_t* __restrict__ indices,      // [B, heads, region, topk]
    scalar_t* __restrict__ grad_input,          // [B, heads, region, w2, c]
    int B, int heads, int region, int topk, int w2, int c) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * heads * region * topk;
    if (idx >= total) return;

    // 反解 grad_output 的多维索引: idx -> (ib, ih, ir, itopk)
    int ik = idx % topk;     int tmp = idx / topk;
    int ir = tmp % region;   tmp /= region;
    int ih = tmp % heads;
    int ib = tmp / heads;

    int64_t src_r = indices[((ib * heads + ih) * region + ir) * topk + ik];
    
    int out_offset = (((ib * heads + ih) * region + ir) * topk + ik) * (w2 * c);
    int in_offset = (((ib * heads + ih) * region + src_r) * w2) * c;
    
    // 将 grad_output 内连续块累加到 grad_input 的相应位置
    for (int i = 0; i < w2 * c; i++) {
        atomicAdd(&grad_input[in_offset + i], grad_output[out_offset + i]);
    }
}

torch::Tensor gather_cuda(torch::Tensor input, torch::Tensor indices) {
    // input: [B, heads, region, w2, c]
    // indices: [B, heads, region, topk]
    auto B = input.size(0);
    auto heads = input.size(1);
    auto region = input.size(2);
    auto w2 = input.size(3);
    auto c = input.size(4);
    int topk = indices.size(3);

    auto output = torch::empty({B, heads, region, topk, w2, c}, input.options());

    int total = B * heads * region * topk;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gather_cuda", ([&] {
        gather_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            B, heads, region, topk, w2, c);
    }));
    return output;
}

torch::Tensor gather_cuda_backward(torch::Tensor grad_output, torch::Tensor indices, std::vector<int64_t> input_shape) {
    // input_shape: [B, heads, region, w2, c]
    auto grad_input = torch::zeros(input_shape, grad_output.options());
    int B = input_shape[0];
    int heads = input_shape[1];
    int region = input_shape[2];
    int w2 = input_shape[3];
    int c = input_shape[4];
    int topk = indices.size(3);

    int total = B * heads * region * topk;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "gather_cuda_backward", ([&] {
        gather_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            grad_input.data_ptr<scalar_t>(),
            B, heads, region, topk, w2, c);
    }));
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_cuda", &gather_cuda, "Custom gather CUDA kernel (forward)");
    m.def("gather_cuda_backward", &gather_cuda_backward, "Custom gather CUDA backward kernel");
}
