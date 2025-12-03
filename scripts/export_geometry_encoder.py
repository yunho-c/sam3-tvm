
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
import mock_setup
from sam3.model.geometry_encoders import SequenceGeometryEncoder, Prompt
import os

import tvm_custom_ops # noqa: F401
from sam3.model.encoder import TransformerEncoderLayer
import math
import torch.nn.functional as F
from manual_attention import ManualMultiheadAttention

# Patch globally
torch.nn.MultiheadAttention = ManualMultiheadAttention

# Replace MultiheadAttention with ManualMultiheadAttention
MultiheadAttention = ManualMultiheadAttention

from sam3.model import geometry_encoders

def patch_concat_padded_sequences():
    """
    Monkeypatch concat_padded_sequences to cast mask to long before summing.
    TVM Relax seems to infer bool sum as bool, causing CodeGenLLVM failure on add(bool, bool).
    """
    original_concat = geometry_encoders.concat_padded_sequences
    
    def concat_padded_sequences_patched(seq1, mask1, seq2, mask2, return_index: bool = False):
        seq1_length, batch_size, hidden_size = seq1.shape
        seq2_length, batch_size, hidden_size = seq2.shape
    
        # assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
        # assert hidden_size == seq1.size(2) == seq2.size(2)
        # assert seq1_length == mask1.size(1)
        # assert seq2_length == mask2.size(1)
    
        # torch._assert_async(is_right_padded(mask1))
        # torch._assert_async(is_right_padded(mask2))
    
        # PATCH: Cast to long before sum
        actual_seq1_lengths = (~mask1).long().sum(dim=-1)
        actual_seq2_lengths = (~mask2).long().sum(dim=-1)
    
        final_lengths = actual_seq1_lengths + actual_seq2_lengths
        max_length = seq1_length + seq2_length
        concatenated_mask = (
            torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1)
            >= final_lengths[:, None]
        )
    
        # (max_len, batch_size, hidden_size)
        concatenated_sequence = torch.zeros(
            (max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype
        )
        concatenated_sequence[:seq1_length, :, :] = seq1
    
        # At this point, the element of seq1 are in the right place
        # We just need to shift the elements of seq2
    
        index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
        index = index + actual_seq1_lengths[None]
    
        concatenated_sequence = concatenated_sequence.scatter(
            0, index[:, :, None].expand(-1, -1, hidden_size), seq2
        )
    
        if return_index:
            return concatenated_sequence, concatenated_mask, index
    
        return concatenated_sequence, concatenated_mask

    geometry_encoders.concat_padded_sequences = concat_padded_sequences_patched
    print("✓ Applied monkeypatch to concat_padded_sequences")

patch_concat_padded_sequences()

def patch_sequence_geometry_encoder():
    """
    Monkeypatch SequenceGeometryEncoder._encode_points to use reshape instead of squeeze.
    TVM Relax has trouble inferring the rank after squeeze(-1) on a 4D tensor from grid_sample,
    causing permute_dims to fail with rank mismatch.
    """
    original_encode_points = SequenceGeometryEncoder._encode_points
    
    def _encode_points_patched(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        n_points, bs = points.shape[:2]

        if self.points_direct_project is not None:
            proj = self.points_direct_project(points)
            assert points_embed is None
            points_embed = proj

        if self.points_pool_project is not None:
            # points are [Num_points, bs, 2], normalized in [0, 1]
            # the grid needs to be [Bs, H_out, W_out, 2] normalized in [-1,1]
            # Will take H_out = num_points, w_out = 1
            grid = points.transpose(0, 1).unsqueeze(2)
            # re normalize to [-1, 1]
            grid = (grid * 2) - 1
            
            # Force grid shape
            grid = grid.view(bs, n_points, 1, 2)
            
            print(f"DEBUG: points.shape={points.shape}")
            print(f"DEBUG: grid.shape={grid.shape}")
            
            # sampled = torch.nn.functional.grid_sample(
            #     img_feats, grid, align_corners=False
            # )
            # Bypass grid_sample to isolate error
            print("DEBUG: Bypassing grid_sample with dummy tensor")
            sampled = torch.zeros(bs, self.d_model, n_points, 1, device=img_feats.device)
            
            # assert list(sampled.shape) == [bs, self.d_model, n_points, 1]
            
            # ORIGINAL: sampled = sampled.squeeze(-1).permute(2, 0, 1)
            # PATCH: Use view to enforce 3D shape explicitly for TVM
            sampled = sampled.view(bs, self.d_model, n_points).permute(2, 0, 1)
            
            proj = self.points_pool_project(sampled)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        if self.points_pos_enc_project is not None:
            x, y = points.unbind(-1)
            enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
            enc_x = enc_x.view(n_points, bs, enc_x.shape[-1])
            enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
            enc = torch.cat([enc_x, enc_y], -1)

            proj = self.points_pos_enc_project(enc)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    SequenceGeometryEncoder._encode_points = _encode_points_patched
    print("✓ Applied monkeypatch to SequenceGeometryEncoder._encode_points")

patch_sequence_geometry_encoder()

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
    return exported_program
    
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
        return mod
            
    except Exception as err:
        print(f"✗ TVM import failed: {err}")
        import traceback
        traceback.print_exc()
        return None

def import_to_tvm(exported_program):
    if exported_program is None:
        return None
    print("\n=== Importing to TVM Relax ===")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ TVM import succeeded")
        
        # Save Relax IR
        with open("sam3_geometry_encoder_tvm.txt", "w") as f:
            f.write(mod.script(show_meta=False))
            
        print("\n=== Inspecting IR for PermuteDims errors ===")
        with open("permute_debug.log", "w") as log_f:
            @relax.expr_functor.visitor
            class PermuteDimsChecker(relax.PyExprVisitor):
                def visit_call_(self, call):
                    if call.op.name == "relax.permute_dims":
                        data = call.args[0]
                        axes = call.attrs.axes
                        if hasattr(data, "struct_info") and isinstance(data.struct_info, relax.TensorStructInfo):
                            ndim = data.struct_info.ndim
                            log_f.write(f"Check: ndim={ndim}, axes={axes}\n")
                            if ndim != len(axes):
                                log_f.write(f"✗ Found invalid permute_dims: ndim={ndim}, axes={axes}\n")
                                log_f.write(f"  Call: {call}\n")
                        else:
                            log_f.write(f"? No struct_info for data in permute_dims: {call}\n")
                    super().visit_call_(call)

            PermuteDimsChecker().visit_expr(mod["main"])
        
        print("\n=== Verifying IR with Normalize ===")
        try:
            mod = relax.transform.Normalize()(mod)
            print("✓ IR verification passed")
        except Exception as e:
            print(f"✗ IR verification failed: {e}")
            raise e
            
        # Apply LegalizeOps to decompose LayerNorm to avoid LLVM debug info issue
        def legalize_layer_norm(bb, call):
            data = call.args[0]
            gamma = call.args[1]
            beta = call.args[2]
            axis = call.attrs.axes
            epsilon = call.attrs.epsilon
            
            # Cast epsilon to float32
            eps = relax.const(epsilon, "float32")
            
            mean = bb.emit(relax.op.mean(data, axis, keepdims=True))
            sub = bb.emit(relax.op.subtract(data, mean))
            sq = bb.emit(relax.op.multiply(sub, sub))
            var = bb.emit(relax.op.mean(sq, axis, keepdims=True))
            std = bb.emit(relax.op.sqrt(relax.op.add(var, eps)))
            out = bb.emit(relax.op.divide(sub, std))
            out = bb.emit(relax.op.multiply(out, gamma))
            out = bb.emit(relax.op.add(out, beta))
            return out

        print("\n=== Applying LayerNorm Legalization ===")
        mod = relax.transform.LegalizeOps({"relax.nn.layer_norm": legalize_layer_norm})(mod)
            
        print("\n=== Compiling with TVM Relax ===")
        ex = relax.build(mod, target="llvm")
        print("✓ Successfully compiled!")
        
        return mod
    except Exception as err:
        print(f"✗ TVM import failed: {err}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    prog = export_geometry_encoder()
    if prog:
        import_to_tvm(prog)
