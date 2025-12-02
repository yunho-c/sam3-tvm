
import torch
import torch.nn as nn
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

import sys
import types
from unittest.mock import MagicMock

# Mock triton and decord as they are not needed for export and might be missing
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
from sam3.model.model_misc import DotProductScoring, MLP

def build_scoring_head():
    """
    Reconstructs the Scoring Head as defined in sam3/model_builder.py
    """
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)

def prepare_dummy_inputs():
    # hs: [num_layer, bs, num_query, d_model]
    num_layer = 6
    bs = 1
    num_query = 10
    d_model = 256
    hs = torch.randn(num_layer, bs, num_query, d_model)
    
    # prompt: [seq, bs, d_model]
    seq = 5
    prompt = torch.randn(seq, bs, d_model)
    
    # prompt_mask: [bs, seq] (1 is valid, 0 is padding)
    # Let's assume all valid for now, or mix
    prompt_mask = torch.ones(bs, seq, dtype=torch.bool) # In code: ~prompt_mask is used. 
    # Code: is_valid = (~prompt_mask).float()...
    # Wait, usually mask=True means padding (ignored).
    # If prompt_mask is True for valid, then ~prompt_mask is False (0).
    # Code: is_valid = (~prompt_mask).float()
    # If prompt_mask is 1 (True), ~prompt_mask is 0 (False).
    # So prompt_mask=True means PADDING?
    # Let's check comment: "prompt_mask has shape (bs, seq), where 1 is valid and 0 is padding"
    # If 1 is valid.
    # ~1 is 0.
    # is_valid = 0.
    # This seems contradictory.
    # Let's check `mean_pool_text`:
    # is_valid = (~prompt_mask).float()
    # If prompt_mask is 1 (valid), ~prompt_mask is 0.
    # Then is_valid is 0.
    # Then pooled_prompt = (prompt * 0).sum() -> 0.
    # This implies prompt_mask=1 means PADDING/INVALID.
    # And prompt_mask=0 means VALID.
    # But comment says "1 is valid and 0 is padding".
    # Maybe `~` on bool tensor flips True/False.
    # If comment is correct, then code `~prompt_mask` would make valid(1) -> 0.
    # This suggests comment might be wrong OR I am misunderstanding `~`.
    # `~` on bool: True->False, False->True.
    # If I want `is_valid` to be 1 for valid tokens.
    # Then `~prompt_mask` should be 1.
    # So `prompt_mask` should be 0 (False) for valid tokens.
    # This aligns with `key_padding_mask` convention in PyTorch (True means padding).
    
    # Let's assume standard PyTorch convention: True = Padding, False = Valid.
    # So prompt_mask = False (0) for valid.
    prompt_mask = torch.zeros(bs, seq, dtype=torch.bool)
    
    return (hs, prompt, prompt_mask)

def export_scoring_head():
    print("=== Building Scoring Head ===")
    scoring_head = build_scoring_head()
    scoring_head.eval()
    
    inputs = prepare_dummy_inputs()
    
    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        outputs = scoring_head(*inputs)
    print("✓ Forward pass succeeded")
    print(f"  scores: {outputs.shape}")
    
    print("\n=== Exporting with torch.export ===")
    try:
        exported_program = torch.export.export(
            scoring_head,
            inputs,
            strict=True
        )
    except Exception as err:
        print(f"✗ Export failed: {err}")
        import traceback
        traceback.print_exc()
        return None
        
    torch.export.save(exported_program, "sam3_scoring_head_exported.pt2")
    print("✓ Exported to sam3_scoring_head_exported.pt2")
    
    print("\n=== Importing to TVM ===")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ TVM import succeeded")
        
        # Basic verification
        print("\nRelax Module:")
        print(mod.script(show_meta=False))
        
        # Save Relax IR
        with open("sam3_scoring_head_tvm.txt", "w") as f:
            f.write(mod.script(show_meta=False))
            
    except Exception as err:
        print(f"✗ TVM import failed: {err}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_scoring_head()
