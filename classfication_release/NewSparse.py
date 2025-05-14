import torch
import SparseAtten

class gatherAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query , key , index , scale):
        attn_weight =SparseAtten.sparse_attention_wrap(query , key , index , scale)
        ctx.save_for_backward(query , key , index)
        ctx.scale = scale
        return attn_weight
    
    @staticmethod
    def backward(ctx, d_atten_weight):
        query , key , index = ctx.saved_tensors
        scale = ctx.scale
        d_query , d_key = SparseAtten.sparse_attention_backward_wrap(d_atten_weight.contiguous() , query , key , index ,scale)

        return d_query , d_key , None , None
    

class AttentionWeighting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atten , value , index , scale):
        
        out = SparseAtten.weighting_forward_warp(atten , value , index , scale)
        ctx.save_for_backward(atten , value , index)
        ctx.scale = scale
        return out
    
    @staticmethod
    def backward(ctx, out):  
        atten , value , index = ctx.saved_tensors 
        scale = ctx.scale
        d_attention , d_value = SparseAtten.weighting_backward_wrap(out.contiguous() , atten , value , index , scale)
        return d_attention , d_value , None , None