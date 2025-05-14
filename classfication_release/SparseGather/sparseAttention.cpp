#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#include <stdio.h>
torch::Tensor sparse_gemm_forward(
    torch::Tensor qpatch,
    torch::Tensor kpatch,
    torch::Tensor topk_index,
    float scale
);

torch::Tensor sparse_gather_forward(
    torch::Tensor qpatch,
    torch::Tensor kpatch,
    torch::Tensor topk_index,
    float scale
){
    CHECK_INPUT(qpatch);
    CHECK_INPUT(kpatch);
    CHECK_INPUT(topk_index);
    return sparse_gemm_forward(qpatch , kpatch , topk_index , scale);

};


std::vector<torch::Tensor> sparse_gemm_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor qpatch,
    torch::Tensor kpatch,
    torch::Tensor r_idx,
    float scale
) ;

std::vector<torch::Tensor> sparse_gather_backward(
    torch::Tensor grad_out,
    torch::Tensor qpatch,
    torch::Tensor kpatch,
    torch::Tensor r_idx,
    float scale
) {
    CHECK_INPUT(qpatch);
    CHECK_INPUT(kpatch);
    CHECK_INPUT(grad_out);
    CHECK_CUDA(r_idx);
    // int B = qpatch.size(0);
    // int heads = qpatch.size(1);
    // int region = qpatch.size(2);
    // int w2 = qpatch.size(3);
    // int c = qpatch.size(4);
    // int topk = r_idx.size(3);
    // printf("start backward g \n");
    // printf("qpatch.shape region*w2 = %d " , qpatch.size(2)*qpatch.size(3));
    return sparse_gemm_backward_cuda(grad_out , qpatch , kpatch , r_idx , scale);
    // printf("end backward g \n");
};


// v forward
torch::Tensor sparse_v_gemm_forward(
    torch::Tensor attn,      // [B, heads, region, topk, w2]
    torch::Tensor vpatch,    // [B, heads, region, w2, c]
    torch::Tensor topk_index,// [B, heads, region, topk]
    int topk
);

torch::Tensor sparse_weighting_forward(
    torch::Tensor attn,      // [B, heads, region, topk, w2]
    torch::Tensor vpatch,    // [B, heads, region, w2, c]
    torch::Tensor topk_index,// [B, heads, region, topk]
    int topk
) {
    CHECK_INPUT(attn);
    CHECK_INPUT(vpatch);
    CHECK_INPUT(topk_index);
    // printf("forward check ! : region*w2 = %d " , attn.size(2)*attn.size(3));
    return sparse_v_gemm_forward(attn , vpatch , topk_index , topk);
};

std::vector<torch::Tensor> sparse_v_gemm_backward_cuda(
    torch::Tensor grad_out,   // [B, heads, region, w2, c]
    torch::Tensor attn,       // [B, heads, region, w2, topk]
    torch::Tensor vpatch,     // [B, heads, region, w2, c]
    torch::Tensor r_idx      // [B, heads, region, topk]
);

std::vector<torch::Tensor> sparse_weighting_backward(
    torch::Tensor grad_out,   // [B, heads, region, w2, c]
    torch::Tensor attn,       // [B, heads, region, w2, topk]
    torch::Tensor vpatch,     // [B, heads, region, w2, c]
    torch::Tensor r_idx      // [B, heads, region, topk]
){
    CHECK_INPUT(grad_out);
    CHECK_INPUT(attn);
    CHECK_INPUT(vpatch);
    CHECK_INPUT(r_idx);
    // printf("start backward v \n"); 
    // printf("grad_out.shape = %d %d %d %d %d \n" , grad_out.size(0),grad_out.size(1),grad_out.size(2),grad_out.size(3),grad_out.size(4));
    // printf("attn.shape = %d %d %d %d %d \n" , attn.size(0),attn.size(1),attn.size(2),attn.size(3),attn.size(4));
    return sparse_v_gemm_backward_cuda(grad_out , attn , vpatch , r_idx);
    // printf(" end backward v \n");
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_gather_forward", &sparse_gather_forward, "Sparse GEMM forward (CUDA)");
    m.def("sparse_gather_backward", &sparse_gather_backward, "Sparse V GEMM forward (CUDA)");
    m.def("sparse_weighting_forward", &sparse_weighting_forward, "Sparse V GEMM backward (CUDA)");
    m.def("sparse_weighting_backward", &sparse_weighting_backward, "Sparse GEMM backward (CUDA)");
}