
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

import mock_setup
from manual_attention import ManualMultiheadAttention

# Import SAM3 components
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.model_misc import MultiheadAttentionWrapper as MultiheadAttention
import math
import torch.nn.functional as F

# Patch globally
torch.nn.MultiheadAttention = ManualMultiheadAttention

# Replace MultiheadAttention
MultiheadAttention = ManualMultiheadAttention

def build_transformer_encoder():
    """
    Reconstructs the Transformer Encoder as defined in sam3/model_builder.py
    """
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=False, # Disable for export
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, 
                src_tensor, src_pos_tensor, 
                prompt, prompt_key_padding_mask):
        
        # Wrap single tensor inputs into lists as expected by the encoder
        # Assuming single feature level for now
        src = [src_tensor]
        src_pos = [src_pos_tensor]
        
        # src_key_padding_mask is optional, we can pass None or a list
        src_key_padding_mask = None
        
        # We pass feat_sizes to avoid the buggy x.dim == 4 check in the else block
        # src_tensor is [L, B, C] where L = h*w
        # We assume h=w=sqrt(L) for simplicity in this export wrapper
        L = src_tensor.shape[0]
        h = int(L**0.5)
        w = h
        assert h * w == L, "L must be a perfect square for this export wrapper"
        feat_sizes = [(h, w)]
        
        outputs = self.encoder(
            src=src,
            prompt=prompt,
            src_key_padding_mask=src_key_padding_mask,
            src_pos=src_pos,
            prompt_key_padding_mask=prompt_key_padding_mask,
            prompt_pos=None, # Optional
            feat_sizes=feat_sizes,
            encoder_extra_kwargs=None
        )
        
        # Return relevant outputs
        # outputs is a dict: "memory", "padding_mask", "pos_embed", "memory_text", ...
        return outputs["memory"], outputs["pos_embed"]

def prepare_dummy_inputs():
    bs = 1
    d_model = 256
    h, w = 32, 32 # Feature map size
    seq_len_src = h * w
    
    # src: [L, B, C] (flattened spatial)
    src_tensor = torch.randn(seq_len_src, bs, d_model)
    
    # src_pos: [L, B, C]
    src_pos_tensor = torch.randn(seq_len_src, bs, d_model)
    
    # prompt: [seq_len, bs, d_model]
    seq_len = 10
    prompt = torch.randn(seq_len, bs, d_model)
    
    # prompt_key_padding_mask: [bs, seq_len]
    prompt_key_padding_mask = torch.zeros(bs, seq_len, dtype=torch.bool)
    
    return (src_tensor, src_pos_tensor, prompt, prompt_key_padding_mask)

def export_transformer_encoder():
    print("=== Building Transformer Encoder ===")
    encoder = build_transformer_encoder()
    wrapper = TransformerEncoderWrapper(encoder)
    wrapper.eval()
    
    inputs = prepare_dummy_inputs()
    
    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        outputs = wrapper(*inputs)
    print("✓ Forward pass succeeded")
    print(f"  memory: {outputs[0].shape}")
    print(f"  pos_embed: {outputs[1].shape}")
    
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
        
    torch.export.save(exported_program, "sam3_transformer_encoder_exported.pt2")
    print("✓ Exported to sam3_transformer_encoder_exported.pt2")
    return exported_program

def import_to_tvm(exported_program):
    if exported_program is None:
        return None
    print("\n=== Importing to TVM ===")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ TVM import succeeded")
        
        # Basic verification
        print("\nRelax Module:")
        print(mod.script(show_meta=False))
        
        # Save Relax IR
        with open("sam3_transformer_encoder_tvm.txt", "w") as f:
            f.write(mod.script(show_meta=False))
            
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
    prog = export_transformer_encoder()
    if prog:
        import_to_tvm(prog)
