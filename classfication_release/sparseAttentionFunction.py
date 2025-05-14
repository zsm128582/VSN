import torch
import sparse_attention

class gatherAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query , key , index , scale):
        attn_weight =sparse_attention.sparse_gather_forward(query , key , index , scale)
        ctx.save_for_backward(query , key , index)
        ctx.scale = scale
        return attn_weight
    
    @staticmethod
    def backward(ctx, d_atten_weight):
        query , key , index = ctx.saved_tensors
        scale = ctx.scale
        d_query , d_key = sparse_attention.sparse_gather_backward(d_atten_weight.contiguous() , query , key , index ,scale)

        return d_query , d_key , None , None
    

class AttentionWeighting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atten , value , index , topk):
        
        out = sparse_attention.sparse_weighting_forward(atten , value , index , topk)
        ctx.save_for_backward(atten , value , index)
        return out
    
    @staticmethod
    def backward(ctx, out):  
        atten , value , index = ctx.saved_tensors 
        d_attention , d_value = sparse_attention.sparse_weighting_backward(out.contiguous() , atten , value , index)
        return d_attention , d_value , None , None