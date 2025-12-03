
import torch
import torch.nn.functional as F
import math
from sam3.model.vitdet import Attention, concat_rel_pos

def manual_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    s = 1 if self.cls_token else 0  # used to exclude cls_token
    if x.ndim == 4:
        B, H, W, _ = x.shape
        assert s == 0  # no cls_token
        L = H * W
        ndim = 4
    else:
        assert x.ndim == 3
        B, L, _ = x.shape
        ndim = 3
        H = W = int(math.sqrt(L - s))

    # qkv with shape (3, B, nHead, L, C)
    qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
    # q, k, v with shape (B, nHead, L, C)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

    # handle rope and rel pos embeddings
    q, k = self._apply_rope(q, k)
    if self.use_rel_pos:
        q, k = concat_rel_pos(
            q.flatten(0, 1),
            k.flatten(0, 1),
            (H, W),
            x.shape[1:3],
            self.rel_pos_h,
            self.rel_pos_w,
            rescale=True,
            relative_coords=self.relative_coords,
        )

        # sdpa expects [B, nheads, H*W, C] so we transpose back
        q = q.reshape(B, self.num_heads, H * W, -1)
        k = k.reshape(B, self.num_heads, H * W, -1)

    # Manual Attention Implementation to avoid SDPA decomposition issues in TVM
    # (specifically the max(bool) check for all-masked rows)
    
    # q, k, v are [B, n_heads, L, head_dim]
    # Scale factor: 1 / sqrt(head_dim)
    # Note: If use_rel_pos is True, head_dim is larger, and SDPA uses the new head_dim.
    # We should match that behavior.
    scale = 1.0 / math.sqrt(q.size(-1))
    
    # [B, n_heads, L, L]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # [B, n_heads, L, head_dim_v]
    # Note: v has original head_dim, not the expanded one from rel_pos
    x = torch.matmul(attn_weights, v)

    if ndim == 4:
        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
    else:
        x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

    x = self.proj(x)

    return x

def apply_patches():
    print("Applying ViTDet Attention patch...")
    Attention.forward = manual_attention_forward
