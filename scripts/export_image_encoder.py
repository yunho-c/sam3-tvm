import torch
import torch.nn as nn
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

import sys
import types
from unittest.mock import MagicMock

# Mock triton as a package
triton = types.ModuleType("triton")
triton.language = MagicMock()
triton.compiler = MagicMock()
triton.jit = MagicMock()
triton.runtime = types.ModuleType("triton.runtime")
triton.runtime.autotuner = MagicMock()
triton.runtime.jit = MagicMock()
triton.runtime.driver = MagicMock()
triton.Config = MagicMock()
triton.__version__ = "2.0.0"

sys.modules["triton"] = triton
sys.modules["triton.language"] = triton.language
sys.modules["triton.compiler"] = triton.compiler
sys.modules["triton.runtime"] = triton.runtime
sys.modules["triton.runtime.autotuner"] = triton.runtime.autotuner
sys.modules["triton.runtime.jit"] = triton.runtime.jit
sys.modules["triton.runtime.driver"] = triton.runtime.driver

sys.modules["decord"] = MagicMock()

# Import SAM3 components
from sam3.model.vitdet import ViT
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine

# Apply RoPE patches
import patch_rope
patch_rope.apply_patches()

# Apply TVM custom ops (scatter, roi_align, floor_divide)
import tvm_custom_ops # noqa: F401

def build_image_encoder():
    """
    Reconstructs the Image Encoder (ViT + Neck) as defined in sam3/model_builder.py
    """
    # Configuration from _create_vit_backbone in sam3/model_builder.py
    # Using smaller values for faster export/testing, but keeping structure
    # Original: img_size=1008, patch_size=14, embed_dim=1024, depth=32, num_heads=16
    # Reduced: img_size=224, patch_size=14, embed_dim=256, depth=2, num_heads=4
    
    # Note: We should try to use close to original config if possible to catch shape issues,
    # but for initial export, reduced size is safer for memory/speed.
    # Let's use a "tiny" version but with all features enabled (RoPE, Window Attn).
    
    img_size = 224 # Must be divisible by patch_size (14) -> 16 patches
    patch_size = 14
    embed_dim = 256
    depth = 2
    num_heads = 4
    window_size = 8 # Must divide img_size/patch_size (16) -> 2 windows
    
    vit = ViT(
        img_size=img_size,
        pretrain_img_size=224, # Match img_size for simplicity
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer="LayerNorm",
        drop_path_rate=0.0, # No dropout for export
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(1,), # Last block is global
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=window_size,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
    )

    # Position encoding
    pos_enc = PositionEmbeddingSine(
        num_pos_feats=256, # Match d_model of neck
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=img_size,
    )

    # Neck
    neck = Sam3DualViTDetNeck(
        position_encoding=pos_enc,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5], # Standard ViTDet scales
        trunk=vit,
        add_sam2_neck=False, # Simplify for now
    )
    
    return neck

class ImageEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        # The neck returns (sam3_out, sam3_pos, sam2_out, sam2_pos)
        # We only care about sam3_out for now (list of tensors)
        # Sam3DualViTDetNeck expects a Tensor, despite the argument name 'tensor_list'
        sam3_out, _, _, _ = self.encoder(x)
        # Return tuple of tensors for export
        return tuple(sam3_out)

def export_image_encoder():
    print("=== Building Image Encoder ===")
    encoder = build_image_encoder()
    wrapper = ImageEncoderWrapper(encoder)
    wrapper.eval()

    # Dummy input
    # Shape: (B, 3, H, W)
    B = 1
    C = 3
    H = 224
    W = 224
    dummy_input = torch.randn(B, C, H, W)

    print("\n=== Exporting to .pt2 ===")
    try:
        exported_program = torch.export.export(
            wrapper,
            (dummy_input,),
            strict=True,
        )
        # Save the exported program
        exported_file = "sam3_image_encoder_exported.pt2"
        torch.export.save(exported_program, exported_file)
        print(f"✓ Exported to {exported_file}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        # Print full traceback if needed
        import traceback
        traceback.print_exc()
        return

    print("\n=== Importing to TVM ===")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        tvm_file = "sam3_image_encoder_tvm.txt"
        with open(tvm_file, "w") as f:
            f.write(str(mod))
        print(f"✓ Imported to TVM and saved to {tvm_file}")
    except Exception as e:
        print(f"✗ TVM import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_image_encoder()
