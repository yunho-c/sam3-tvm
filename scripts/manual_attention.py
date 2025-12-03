
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ManualMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
            if bias:
                self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
            else:
                self.register_parameter('in_proj_bias', None)
        else:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim)))
            if bias:
                self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
            
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
        print(f"DEBUG: Shared ManualMultiheadAttention initialized with embed_dim={embed_dim}, num_heads={num_heads}")

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else (None, None, None)
        
        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)
        
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_mask = torch.zeros(attn_mask.shape, dtype=scores.dtype, device=attn_mask.device)
                new_mask.masked_fill_(attn_mask, -1e9)
                attn_mask = new_mask
            
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, src_len)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                mask_float = torch.zeros(key_padding_mask.shape, dtype=scores.dtype, device=key_padding_mask.device)
                mask_float.masked_fill_(key_padding_mask, -1e9)
                key_padding_mask = mask_float
                
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores + key_padding_mask
            
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)
        output = output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        output = self.out_proj(output)
        
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, attn_probs
