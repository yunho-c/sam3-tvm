
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
from sam3.model.maskformer_segmentation import UniversalSegmentationHead, PixelDecoder
from sam3.model.model_misc import MultiheadAttentionWrapper as MultiheadAttention

def build_segmentation_head():
    """
    Reconstructs the Segmentation Head as defined in sam3/model_builder.py
    """
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=None,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=False, # Disable checkpointing for export
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head

class SegmentationHeadWrapper(nn.Module):
    def __init__(self, seg_head):
        super().__init__()
        self.seg_head = seg_head

    def forward(self, 
                feat0: torch.Tensor, 
                feat1: torch.Tensor, 
                feat2: torch.Tensor, 
                feat3: torch.Tensor, 
                obj_queries: torch.Tensor, 
                encoder_hidden_states: torch.Tensor,
                prompt: torch.Tensor,
                prompt_mask: torch.Tensor):
        
        backbone_feats = [feat0, feat1, feat2, feat3]
        
        # Assuming batch size 1 for export
        bs = obj_queries.shape[0]
        image_ids = torch.zeros(bs, dtype=torch.long, device=obj_queries.device)
        
        outputs = self.seg_head(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask
        )
        
        # Return tuple for export
        return outputs["pred_masks"], outputs["semantic_seg"]

def prepare_dummy_inputs():
    # Based on standard SAM/ViTDet shapes
    # Backbone features: [B, C, H, W]
    # C=256 (hidden_dim)
    # Shapes typically: 1/4, 1/8, 1/16, 1/32 of input resolution (1008x1008)
    # So: 252x252, 126x126, 63x63, 31x31 (approx)
    # Let's use smaller shapes for testing/export to be safe and fast, 
    # but consistent with upsampling stages (3 stages implies 2^3=8 factor relation?)
    # PixelDecoder upsamples.
    
    # Let's use 64x64 base and downsample
    # feat0: 64x64
    # feat1: 32x32
    # feat2: 16x16
    # feat3: 8x8
    
    bs = 1
    c = 256
    
    feat0 = torch.randn(bs, c, 64, 64)
    feat1 = torch.randn(bs, c, 32, 32)
    feat2 = torch.randn(bs, c, 16, 16)
    feat3 = torch.randn(bs, c, 8, 8)
    
    # obj_queries: [L, B, NumQueries, C]
    # TransformerDecoder outputs intermediate queries from each layer.
    # UniversalSegmentationHead uses obj_queries[-1] (last layer).
    num_layers = 2
    num_queries = 10
    obj_queries = torch.randn(num_layers, bs, num_queries, c)
    
    # encoder_hidden_states: [B, S, C]
    # S is spatial dim of last feat? Or combined?
    # In _embed_pixels: encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
    # spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
    # encoder_visual_embed = encoder_hidden_states[..., :spatial_dim]
    # So encoder_hidden_states must have enough tokens to cover the last feature map spatial dim.
    # feat3 is 8x8 = 64 tokens.
    # Let's make it larger to be safe, e.g. 100
    s = 64 # Must match feat3 spatial dim (8*8) if code relies on it?
    # Code: encoder_visual_embed = encoder_hidden_states[..., :spatial_dim]
    # So S >= spatial_dim
    # Format: [S, B, C] (batch_first=False)
    encoder_hidden_states = torch.randn(100, bs, c) 
    
    # Prompt: [NumPrompts, B, C] (batch_first=False)
    num_prompts = 5
    prompt = torch.randn(num_prompts, bs, c)
    # Prompt mask: [B, NumPrompts] (always batch first for mask)
    prompt_mask = torch.zeros(bs, num_prompts, dtype=torch.bool)
    
    return (feat0, feat1, feat2, feat3, obj_queries, encoder_hidden_states, prompt, prompt_mask)

def export_segmentation_head():
    print("=== Building Segmentation Head ===")
    seg_head = build_segmentation_head()
    wrapper = SegmentationHeadWrapper(seg_head)
    wrapper.eval()
    
    inputs = prepare_dummy_inputs()
    
    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        outputs = wrapper(*inputs)
    print("✓ Forward pass succeeded")
    print(f"  pred_masks: {outputs[0].shape}")
    print(f"  semantic_seg: {outputs[1].shape}")
    
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
        
    torch.export.save(exported_program, "sam3_segmentation_head_exported.pt2")
    print("✓ Exported to sam3_segmentation_head_exported.pt2")
    
    print("\n=== Importing to TVM ===")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ TVM import succeeded")
        
        # Basic verification
        print("\nRelax Module:")
        print(mod.script(show_meta=False))
        
        # Save Relax IR
        with open("sam3_segmentation_head_tvm.txt", "w") as f:
            f.write(mod.script(show_meta=False))
            
    except Exception as err:
        print(f"✗ TVM import failed: {err}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_segmentation_head()
