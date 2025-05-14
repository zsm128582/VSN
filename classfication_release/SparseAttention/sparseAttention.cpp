#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#include <stdio.h>
torch::Tensor sparse_attention(
    torch::Tensor query,  // [B, H, R, W2, C]
    torch::Tensor key,    // [B, H, R, W2, C]
    torch::Tensor idx,    // [B, H, R, K]
    float scale
);

torch::Tensor sparse_attention_wrap(
    torch::Tensor qpatch,
    torch::Tensor kpatch,
    torch::Tensor topk_index,
    float scale
){
    CHECK_INPUT(qpatch);
    CHECK_INPUT(kpatch);
    CHECK_INPUT(topk_index);
    return sparse_attention(qpatch , kpatch , topk_index , scale);

};


std::vector<torch::Tensor> sparse_attention_backward(
    torch::Tensor d_atten,   // [B, H, R, W2, K, W2]
    torch::Tensor query,     // [B, H, R, W2, C]
    torch::Tensor key,       // [B, H, R, W2, C]
    torch::Tensor idx,       // [B, H, R, K]
    float scale             // 缩放因子
) ;

std::vector<torch::Tensor> sparse_attention_backward_wrap(
    torch::Tensor d_atten,   // [B, H, R, W2, K, W2]
    torch::Tensor query,     // [B, H, R, W2, C]
    torch::Tensor key,       // [B, H, R, W2, C]
    torch::Tensor idx,       // [B, H, R, K]
    float scale             // 缩放因子
)  
{
    CHECK_INPUT(query);
    CHECK_INPUT(key);
    CHECK_INPUT(d_atten);
    CHECK_CUDA(idx);
    // int B = qpatch.size(0);
    // int heads = qpatch.size(1);
    // int region = qpatch.size(2);
    // int w2 = qpatch.size(3);
    // int c = qpatch.size(4);
    // int topk = r_idx.size(3);
    // printf("start backward g \n");
    // printf("qpatch.shape region*w2 = %d " , qpatch.size(2)*qpatch.size(3));
    return sparse_attention_backward(d_atten , query , key , idx , scale);
    // printf("end backward g \n");
};

torch::Tensor weighting_forward(
    torch::Tensor attn,  // [B, H, R, W2, K , W2 ]
    torch::Tensor value,    // [B, H, R, W2, C]
    torch::Tensor idx,    // [B, H, R, K]
    float scale
);

torch::Tensor weighting_forward_warp(
    torch::Tensor attn,  // [B, H, R, W2, K , W2 ]
    torch::Tensor value,    // [B, H, R, W2, C]
    torch::Tensor idx,    // [B, H, R, K]
    float scale
){
    CHECK_INPUT(attn);
    CHECK_INPUT(value);
    CHECK_INPUT(idx);
    return weighting_forward(attn , value , idx , scale);
};

std::vector<torch::Tensor> sparse_weighting_backward(
    torch::Tensor d_out,   // [B, H, R, W2, C]
    torch::Tensor atten,     // [B, H, R, W2,  K , W2]
    torch::Tensor value,       // [B, H, R, W2, C]
    torch::Tensor idx,       // [B, H, R, K]
    float scale             // 缩放因子
);


std::vector<torch::Tensor> weighting_backward_wrap(
    torch::Tensor d_out,   // [B, H, R, W2, C]
    torch::Tensor atten,     // [B, H, R, W2,  K , W2]
    torch::Tensor value,       // [B, H, R, W2, C]
    torch::Tensor idx,       // [B, H, R, K]
    float scale             // 缩放因子
){
    CHECK_INPUT(d_out);
    CHECK_INPUT(atten);
    CHECK_INPUT(value);
    CHECK_CUDA(idx);
    return sparse_weighting_backward(d_out , atten , value , idx , scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention_wrap", &sparse_attention_wrap, "Sparse GEMM forward (CUDA)");
    m.def("sparse_attention_backward_wrap", &sparse_attention_backward_wrap, "Sparse V GEMM forward (CUDA)");
    m.def("weighting_forward_warp", &weighting_forward_warp, "Sparse V GEMM forward (CUDA)");
    m.def("weighting_backward_wrap", &weighting_backward_wrap, "Sparse V GEMM forward (CUDA)");
}