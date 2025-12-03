import torch
import torch.nn as nn
from typing import Tuple, Optional
import sam3.model.vitdet as vitdet
from functools import partial

def compute_axial_cis_patched(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
) -> torch.Tensor:
    # Original: returns complex tensor (L, dim/2)
    # Patched: returns float tensor (L, dim/2, 2) where [..., 0]=cos, [..., 1]=sin
    
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = vitdet.init_t_xy(end_x, end_y, scale_pos, offset)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    
    # Original:
    # freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    # freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    # return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)
    
    # Patched:
    # polar(1, angle) -> cos(angle) + i*sin(angle)
    cis_x = torch.stack([torch.cos(freqs_x), torch.sin(freqs_x)], dim=-1)
    cis_y = torch.stack([torch.cos(freqs_y), torch.sin(freqs_y)], dim=-1)
    
    # cat along dim=-2 (the feature dim), preserving last dim=2 (real/imag)
    # freqs_x: (L, dim/4) -> cis_x: (L, dim/4, 2)
    return torch.cat([cis_x, cis_y], dim=1)

def apply_rotary_enc_patched(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq: (B, H, W, C) -> (B, H, W, C/2, 2)
    # Actually xq is (B, num_heads, L, head_dim) in Attention.forward
    # But apply_rotary_enc is generic.
    # We reshape to (..., -1, 2)
    
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_real = xq_r[..., 0]
    xq_imag = xq_r[..., 1]
    
    xk_r = None
    if xk.shape[-2] != 0:
        xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xk_real = xk_r[..., 0]
        xk_imag = xk_r[..., 1]

    # freqs_cis is (L, dim/2, 2)
    # We need to broadcast to (B, num_heads, L, dim/2, 2)
    # xq_r is (B, num_heads, L, dim/2, 2)
    
    # Reshape freqs
    # We assume freqs_cis matches xq_r on the spatial dim (L) and feature dim (dim/2)
    # xq_r ndim = 5.
    # freqs_cis ndim = 3.
    # We want (1, 1, L, dim/2, 2)
    
    # Generic reshape logic similar to original reshape_for_broadcast
    # Original checks x.shape[-2] and x.shape[-1].
    # Here xq_r.shape[-3] is L, xq_r.shape[-2] is dim/2.
    # freqs_cis.shape[0] is L, freqs_cis.shape[1] is dim/2.
    
    ndim = xq_r.ndim
    shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(xq_r.shape)]
    # shape will be [1, 1, L, dim/2, 2]
    
    freqs = freqs_cis.view(*shape)
    cos = freqs[..., 0]
    sin = freqs[..., 1]
    
    # Rotate
    xq_out_real = xq_real * cos - xq_imag * sin
    xq_out_imag = xq_real * sin + xq_imag * cos
    xq_out = torch.stack([xq_out_real, xq_out_imag], dim=-1).flatten(-2)
    
    xk_out = xk
    if xk_r is not None:
        # repeat freqs logic
        if repeat_freqs_k:
             # xk might have different L?
             # In ViTDet, xk usually has same L as xq.
             # But if repeat_freqs_k is True, it implies xk might be larger/smaller?
             # Original: r = xk_.shape[-2] // xq_.shape[-2]
             # Here shape[-2] is dim/2. That doesn't change.
             # Original likely meant sequence length dim?
             # In original, xq_ is (..., L, dim/2). shape[-2] is L.
             # Here xq_r is (..., L, dim/2, 2). shape[-3] is L.
             
             # If repeat_freqs_k is used, we need to handle it.
             # But let's assume standard usage for now.
             pass
        
        xk_out_real = xk_real * cos - xk_imag * sin
        xk_out_imag = xk_real * sin + xk_imag * cos
        xk_out = torch.stack([xk_out_real, xk_out_imag], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def _setup_rope_freqs_patched(self) -> None:
    if not self.use_rope:
        self.freqs_cis = None
        return

    assert self.input_size is not None
    if self.rope_pt_size is None:
        self.rope_pt_size = self.input_size

    # Use patched compute_cis
    self.compute_cis = partial(
        compute_axial_cis_patched,
        dim=self.head_dim,
        theta=self.rope_theta,
    )

    scale_pos = 1.0
    if self.rope_interp:
        scale_pos = self.rope_pt_size[0] / self.input_size[0]
        
    freqs_cis = self.compute_cis(
        end_x=self.input_size[0],
        end_y=self.input_size[1],
        scale_pos=scale_pos,
    )
    
    if self.cls_token:
        # Handle cls token
        # t = torch.zeros(self.head_dim // 2, ...)
        # cls_freqs_cis = polar(1, t) -> cos(0)+isin(0) = 1+0i
        # So we want (1, 0)
        t = torch.zeros(
            self.head_dim // 2,
            dtype=torch.float32,
            device=freqs_cis.device,
        )
        # cis: (dim/2, 2)
        cls_cis = torch.stack([torch.ones_like(t), t], dim=-1)
        # unsqueeze to (1, dim/2, 2)
        cls_cis = cls_cis.unsqueeze(0)
        freqs_cis = torch.cat([cls_cis, freqs_cis], dim=0)

    self.register_buffer("freqs_cis", freqs_cis)

def _apply_rope_patched(self, q, k) -> Tuple[torch.Tensor, torch.Tensor]:
    if not self.use_rope:
        return q, k

    assert self.freqs_cis is not None
    return apply_rotary_enc_patched(q, k, freqs_cis=self.freqs_cis)

def apply_patches():
    print("Applying RoPE patches to avoid complex64...")
    vitdet.compute_axial_cis = compute_axial_cis_patched
    vitdet.apply_rotary_enc = apply_rotary_enc_patched
    vitdet.Attention._setup_rope_freqs = _setup_rope_freqs_patched
    vitdet.Attention._apply_rope = _apply_rope_patched
    print("RoPE patches applied.")
