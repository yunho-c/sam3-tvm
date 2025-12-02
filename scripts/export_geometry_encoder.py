
import torch
import torch.nn as nn
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

import sys
import types
from unittest.mock import MagicMock

# Mock triton and decord
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
# Import SAM3 components
from sam3.model.geometry_encoders import SequenceGeometryEncoder, Prompt
from sam3.model.model_misc import MultiheadAttentionWrapper as MultiheadAttention
from sam3.model.encoder import TransformerEncoderLayer

def build_geometry_encoder():
    """
    Reconstructs the Geometry Encoder as defined in sam3/model_builder.py
    """
    # Default config from model_builder.py (inferred)
    # _create_image_encoder -> _create_geometry_encoder
    
    # We need a position encoding module
    class PositionEmbeddingSine(nn.Module):
        def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
            super().__init__()
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * 3.141592653589793
            self.scale = scale

        def _encode_xy(self, x, y):
            # x, y are flattened
            assert x.dim() == 1
            y_embed = y * self.scale
            x_embed = x * self.scale
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats)

            pos_x = x_embed[:, None] / dim_t
            pos_y = y_embed[:, None] / dim_t
            pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
            pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
            return pos_x, pos_y

        def encode_boxes(self, cx, cy, w, h):
            pos_x, pos_y = self._encode_xy(cx, cy)
            pos_w, pos_h = self._encode_xy(w, h)
            return torch.cat([pos_y, pos_x, pos_h, pos_w], dim=1)

    pos_enc = PositionEmbeddingSine(num_pos_feats=128, normalize=True)

    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    encoder = SequenceGeometryEncoder(
        encode_boxes_as_points=True,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=False,
        boxes_pool=True,
        boxes_pos_enc=False,
        d_model=256,
        pos_enc=pos_enc,
        num_layers=2,
        layer=geo_layer,
        roi_size=7,
        add_cls=True,
        add_post_encode_proj=True,
        mask_encoder=None, # Simplifying for now
        add_mask_label=False,
    )
    return encoder

class GeometryEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, 
                point_embeddings, point_mask, point_labels,
                box_embeddings, box_mask, box_labels,
                img_feats_last, img_size_h, img_size_w):
        
        # Reconstruct Prompt object
        # Note: Prompt object is just a container, but SequenceGeometryEncoder expects it.
        # We can't pass the object directly to exported graph, so we pass tensors and reconstruct inside.
        
        # However, SequenceGeometryEncoder.forward takes `geo_prompt: Prompt`.
        # torch.export might trace through the Prompt object if it's just a container.
        # But constructing it inside forward is safer for export signature.
        
        geo_prompt = Prompt(
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels,
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            mask_embeddings=None,
            mask_mask=None,
            mask_labels=None
        )
        
        # img_feats is expected to be a list, but we only use the last one for pooling
        # SequenceGeometryEncoder: seq_first_img_feats = img_feats[-1]
        img_feats = [img_feats_last]
        
        # img_sizes is list of (H, W) tuples
        img_sizes = [(img_size_h, img_size_w)]
        
        return self.encoder(geo_prompt, img_feats, img_sizes)

def prepare_dummy_inputs():
    bs = 1
    d_model = 256
    
    # Points
    n_points = 5
    point_embeddings = torch.rand(n_points, bs, 2) # Normalized [0, 1]
    point_mask = torch.zeros(bs, n_points, dtype=torch.bool)
    point_labels = torch.ones(n_points, bs, dtype=torch.long)
    
    # Boxes
    n_boxes = 2
    box_embeddings = torch.rand(n_boxes, bs, 4) # Normalized CxCyWH
    box_mask = torch.zeros(bs, n_boxes, dtype=torch.bool)
    box_labels = torch.ones(n_boxes, bs, dtype=torch.long)
    
    # Image features for pooling
    # Shape: [H*W, B, C] (sequence first)
    H, W = 64, 64
    img_feats_last = torch.randn(H*W, bs, d_model)
    
    return (point_embeddings, point_mask, point_labels,
            box_embeddings, box_mask, box_labels,
            img_feats_last, H, W)

def export_geometry_encoder():
    print("=== Building Geometry Encoder ===")
    encoder = build_geometry_encoder()
    wrapper = GeometryEncoderWrapper(encoder)
    wrapper.eval()
    
    inputs = prepare_dummy_inputs()
    
    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        outputs = wrapper(*inputs)
    print("✓ Forward pass succeeded")
    # Output is (final_embeds, final_mask)
    print(f"  embeds: {outputs[0].shape}")
    print(f"  mask: {outputs[1].shape}")
    
    print("\n=== Exporting with torch.export ===")
    try:
        exported_program = torch.export.export(
            wrapper,
            inputs,
            strict=True
        )
    except Exception as err:
        print(f"✗ Export failed: {err}")
        import traceback
        traceback.print_exc()
        return None
        
    torch.export.save(exported_program, "sam3_geometry_encoder_exported.pt2")
    print("✓ Exported to sam3_geometry_encoder_exported.pt2")
    
    print("\n=== Importing to TVM ===")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ TVM import succeeded")
        
        # Basic verification
        print("\nRelax Module:")
        print(mod.script(show_meta=False))
        
        # Save Relax IR
        with open("sam3_geometry_encoder_tvm.txt", "w") as f:
            f.write(mod.script(show_meta=False))
            
    except Exception as err:
        print(f"✗ TVM import failed: {err}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_geometry_encoder()
