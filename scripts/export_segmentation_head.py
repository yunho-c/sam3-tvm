
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import mock_setup
import tvm_custom_ops
from manual_attention import ManualMultiheadAttention

# Patch globally
torch.nn.MultiheadAttention = ManualMultiheadAttention

# Replace MultiheadAttention
MultiheadAttention = ManualMultiheadAttention

# Import SAM3 components
from sam3.model.maskformer_segmentation import UniversalSegmentationHead, PixelDecoder

# ... (rest of file)

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
        with open("sam3_segmentation_head_tvm.txt", "w") as f:
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

# Monkeypatch PixelDecoder.forward to use scale_factor instead of size
# This avoids dynamic size calculation which causes issues with TVM resize2d expecting PrimExpr
def pixel_decoder_forward_patched(self, backbone_feats):
    # Assumes backbone features are already projected (C == hidden dim)

    prev_fpn = backbone_feats[-1]
    fpn_feats = backbone_feats[:-1]
    for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
        curr_fpn = bb_feat
        # Use scale_factor=2.0 instead of size=curr_fpn.shape[-2:]
        # SAM backbone features are typically 1/32, 1/16, 1/8, 1/4, so 2x upsample is correct.
        prev_fpn = curr_fpn + F.interpolate(
            prev_fpn, scale_factor=2.0, mode=self.interpolation_mode
        )
        if self.shared_conv:
            # only one conv layer
            layer_idx = 0
        prev_fpn = self.conv_layers[layer_idx](prev_fpn)
        prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

    return prev_fpn

PixelDecoder.forward = pixel_decoder_forward_patched

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
    
    # ... (rest of function)
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
    
    # Define dynamic shapes
    # We use static spatial shapes to avoid resize2d issues and constraint violations
    # Input 1024x1024
    # feat0: 256x256
    # feat1: 128x128
    # feat2: 64x64
    # feat3: 32x32
    
    B = 1 # Static batch size
    
    # Static spatial dims
    H0_val, W0_val = 256, 256
    H1_val, W1_val = 128, 128
    H2_val, W2_val = 64, 64
    H3_val, W3_val = 32, 32
    
    # S must match feat3 spatial dim (32*32 = 1024)
    S_val = 1024
    
    # NQ is fixed usually
    NQ_val = 10 # or whatever dummy input uses
    
    # P (num_prompts) can be dynamic?
    # Let's try making P dynamic, everything else static.
    P = torch.export.Dim("num_prompts", min=1)
    
    # Re-create dummy inputs with these shapes to be sure
    # prepare_dummy_inputs uses hardcoded shapes. We should update them or just rely on them being compatible?
    # prepare_dummy_inputs used:
    # feat0: 64x64 -> We need 256x256
    # So we must update prepare_dummy_inputs or re-generate inputs here.
    
    print("Regenerating inputs for static export...")
    c = 256
    feat0 = torch.randn(B, c, H0_val, W0_val)
    feat1 = torch.randn(B, c, H1_val, W1_val)
    feat2 = torch.randn(B, c, H2_val, W2_val)
    feat3 = torch.randn(B, c, H3_val, W3_val)
    
    num_layers = 2
    num_queries = 10 # NQ
    obj_queries = torch.randn(num_layers, B, num_queries, c)
    
    encoder_hidden_states = torch.randn(S_val, B, c)
    
    num_prompts = 5
    prompt = torch.randn(num_prompts, B, c)
    prompt_mask = torch.zeros(B, num_prompts, dtype=torch.bool)
    
    inputs = (feat0, feat1, feat2, feat3, obj_queries, encoder_hidden_states, prompt, prompt_mask)
    
    dynamic_shapes = (
        None, # feat0 static
        None, # feat1 static
        None, # feat2 static
        None, # feat3 static
        None, # obj_queries static
        None, # encoder_hidden_states static
        {0: P}, # prompt dynamic P
        {1: P}, # prompt_mask dynamic P
    )

    try:
        exported_program = torch.export.export(
            wrapper,
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=True
        )
    except Exception as err:
        print(f"✗ Export failed: {err}")
        import traceback
        traceback.print_exc()
        return None
        
    torch.export.save(exported_program, "sam3_segmentation_head_exported.pt2")
    print("✓ Exported to sam3_segmentation_head_exported.pt2")
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
        with open("sam3_segmentation_head_tvm.txt", "w") as f:
            f.write(mod.script(show_meta=False))
        return mod
            
    except Exception as err:
        print(f"✗ TVM import failed: {err}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    prog = export_segmentation_head()
    if prog:
        import_to_tvm(prog)
