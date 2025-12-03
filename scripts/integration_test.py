
import torch
import tvm
from tvm import relax
import numpy as np
import sys
import os

# Add scripts directory to path to import export modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import export_vision_backbone
import export_geometry_encoder
import export_transformer_encoder
import export_decoder
import export_segmentation_head
import export_scoring_head

import patch_rope
import tvm_custom_ops

def compile_module(mod, target="llvm"):
    """Compile a Relax module to a TVM VM."""
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm

def to_numpy(t):
    return t.detach().cpu().numpy()

def main():
    print("=== SAM3 TVM Integration Test ===")
    
    # 1. Export and Import all components
    print("\n[1/6] Loading Vision Backbone...")
    prog_vision = export_vision_backbone.export_with_dynamo()
    mod_vision = export_vision_backbone.import_to_tvm(prog_vision)
    vm_vision = compile_module(mod_vision)
    
    print("\n[2/6] Loading Geometry Encoder...")
    prog_geo = export_geometry_encoder.export_geometry_encoder()
    mod_geo = export_geometry_encoder.import_to_tvm(prog_geo)
    vm_geo = compile_module(mod_geo)
    
    print("\n[3/6] Loading Transformer Encoder...")
    prog_enc = export_transformer_encoder.export_transformer_encoder()
    mod_enc = export_transformer_encoder.import_to_tvm(prog_enc)
    vm_enc = compile_module(mod_enc)
    
    print("\n[4/6] Loading Transformer Decoder...")
    prog_dec = export_decoder.export_decoder()
    mod_dec = export_decoder.import_to_tvm(prog_dec)
    vm_dec = compile_module(mod_dec)
    
    print("\n[5/6] Loading Segmentation Head...")
    prog_seg = export_segmentation_head.export_segmentation_head()
    mod_seg = export_segmentation_head.import_to_tvm(prog_seg)
    vm_seg = compile_module(mod_seg)
    
    print("\n[6/6] Loading Scoring Head...")
    prog_score = export_scoring_head.export_scoring_head()
    mod_score = export_scoring_head.import_to_tvm(prog_score)
    vm_score = compile_module(mod_score)
    
    print("\n=== Running End-to-End Pipeline ===")
    
    # --- Step 1: Vision Backbone ---
    print("Running Vision Backbone...")
    # Input: (1, 3, 1008, 1008)
    img = torch.randn(1, 3, 1008, 1008)
    img_tvm = tvm.nd.array(to_numpy(img))
    
    # Output: tuple(feat0, feat1, feat2, feat3, pos0, pos1, pos2, pos3)
    # feat3 is 1/32 scale -> 31x31 approx (1008/32 = 31.5) -> 32x32 actually?
    # 1008 / 16 = 63.
    # Let's check shapes from output.
    vision_out = vm_vision["main"](img_tvm)
    
    # Unpack
    # vision_out is a TVM Tuple
    feats = [vision_out[i] for i in range(4)]
    pos_embeds = [vision_out[i+4] for i in range(4)]
    
    feat3 = feats[3] # Shape: [1, 256, 32, 32] (assuming 32x32)
    pos3 = pos_embeds[3]
    
    print(f"  feat3 shape: {feat3.shape}")
    
    # --- Step 2: Geometry Encoder ---
    print("Running Geometry Encoder...")
    # Inputs: points, boxes, img_feats_last, img_size
    bs = 1
    n_points = 5
    n_boxes = 2
    
    point_embeddings = torch.rand(n_points, bs, 2)
    point_mask = torch.zeros(bs, n_points, dtype=torch.bool)
    point_labels = torch.ones(n_points, bs, dtype=torch.long)
    
    box_embeddings = torch.rand(n_boxes, bs, 4)
    box_mask = torch.zeros(bs, n_boxes, dtype=torch.bool)
    box_labels = torch.ones(n_boxes, bs, dtype=torch.long)
    
    # Prepare img_feats_last: Flatten feat3 [B, C, H, W] -> [H*W, B, C]
    # Note: feat3 is TVM NDArray. Convert to torch/numpy to reshape?
    # Or use TVM reshape. Let's use numpy for glue logic simplicity.
    feat3_np = feat3.numpy() # [1, 256, 32, 32]
    B, C, H, W = feat3_np.shape
    # Permute to [H, W, B, C] then flatten to [H*W, B, C]
    # Wait, check export_geometry_encoder expectation.
    # It expects: img_feats_last = torch.randn(H*W, bs, d_model)
    # And typically sequence is H*W.
    # PyTorch: x.flatten(2).permute(2, 0, 1) -> [B, C, H*W] -> [H*W, B, C]
    feat3_flat = feat3_np.reshape(B, C, H*W).transpose(2, 0, 1) # [H*W, B, C]
    
    img_size_h = np.array(H, dtype=np.int64) # Scalar? TVM expects args.
    img_size_w = np.array(W, dtype=np.int64)
    
    geo_inputs = [
        tvm.nd.array(to_numpy(point_embeddings)),
        tvm.nd.array(to_numpy(point_mask)),
        tvm.nd.array(to_numpy(point_labels)),
        tvm.nd.array(to_numpy(box_embeddings)),
        tvm.nd.array(to_numpy(box_mask)),
        tvm.nd.array(to_numpy(box_labels)),
        tvm.nd.array(feat3_flat),
        tvm.nd.array(img_size_h), # Pass as 0-rank array or scalar?
        tvm.nd.array(img_size_w)
    ]
    
    # Note: Scalar inputs in Relax VM might need to be passed as NDArray if defined as Tensor in Relax.
    # In export_geometry_encoder, img_size_h/w were passed as int to wrapper?
    # Wrapper signature: forward(..., img_size_h, img_size_w)
    # If they were traced as SymInt or Tensor, it matters.
    # In export_geometry_encoder.py: img_sizes = [(img_size_h, img_size_w)]
    # It seems they are treated as values.
    # Let's try passing as NDArray first.
    
    geo_out = vm_geo["main"](*geo_inputs)
    prompt_embeds = geo_out[0]
    prompt_mask = geo_out[1]
    
    print(f"  prompt_embeds shape: {prompt_embeds.shape}")
    
    # --- Step 3: Transformer Encoder ---
    print("Running Transformer Encoder...")
    # Inputs: src, src_pos, prompt, prompt_mask
    # src: feat3_flat
    # src_pos: pos3 flattened similarly
    pos3_np = pos3.numpy()
    pos3_flat = pos3_np.reshape(B, C, H*W).transpose(2, 0, 1)
    
    enc_inputs = [
        tvm.nd.array(feat3_flat),
        tvm.nd.array(pos3_flat),
        prompt_embeds,
        prompt_mask
    ]
    
    enc_out = vm_enc["main"](*enc_inputs)
    memory = enc_out[0]
    pos_embed = enc_out[1]
    
    print(f"  memory shape: {memory.shape}")
    
    # --- Step 4: Transformer Decoder ---
    print("Running Transformer Decoder...")
    # Inputs: tgt, memory, pos, ref_boxes, level_start_index, valid_ratios, text_mem, text_mask
    
    # Prepare dummy inputs for decoder specific ones
    num_queries = 300
    d_model = 256
    tgt = torch.randn(num_queries, B, d_model)
    ref_boxes = torch.rand(num_queries, B, 4)
    level_start_index = torch.tensor([0], dtype=torch.long)
    valid_ratios = torch.ones(B, 1, 2)
    
    # Dummy text inputs
    num_text = 16
    text_mem = torch.randn(num_text, B, d_model)
    text_mask = torch.zeros(B, num_text, dtype=torch.bool)
    
    dec_inputs = [
        tvm.nd.array(to_numpy(tgt)),
        memory,
        pos_embed,
        tvm.nd.array(to_numpy(ref_boxes)),
        tvm.nd.array(to_numpy(level_start_index)),
        tvm.nd.array(to_numpy(valid_ratios)),
        tvm.nd.array(to_numpy(text_mem)),
        tvm.nd.array(to_numpy(text_mask))
    ]
    
    dec_out = vm_dec["main"](*dec_inputs)
    # Output: (intermediate, ref_boxes, ...)
    intermediate = dec_out[0] # List of intermediates? Or tensor [L, B, NQ, C]?
    # In export_decoder.py, it returns outputs[0] which is intermediate.
    # Shape: [num_layers, bs, num_queries, d_model]
    
    print(f"  intermediate shape: {intermediate.shape}")
    
    # --- Step 5: Segmentation Head ---
    print("Running Segmentation Head...")
    # Inputs: feat0..3, obj_queries, encoder_hidden_states, prompt, prompt_mask
    # obj_queries: intermediate[-1] -> slice it
    # intermediate is [L, B, NQ, C].
    # We need to pass the whole intermediate tensor if the head expects it?
    # export_segmentation_head.py: obj_queries input is [L, B, NQ, C]
    
    # encoder_hidden_states: memory [S, B, C]
    # prompt, prompt_mask: from geo encoder
    
    seg_inputs = [
        feats[0], feats[1], feats[2], feats[3],
        intermediate,
        memory,
        prompt_embeds,
        prompt_mask
    ]
    
    seg_out = vm_seg["main"](*seg_inputs)
    pred_masks = seg_out[0]
    semantic_seg = seg_out[1]
    
    print(f"  pred_masks shape: {pred_masks.shape}")
    
    # --- Step 6: Scoring Head ---
    print("Running Scoring Head...")
    # Inputs: hs (intermediate), prompt, prompt_mask
    
    score_inputs = [
        intermediate,
        prompt_embeds,
        prompt_mask
    ]
    
    score_out = vm_score["main"](*score_inputs)
    scores = score_out
    
    print(f"  scores shape: {scores.shape}")
    
    print("\n✓ End-to-End Pipeline Successful!")

if __name__ == "__main__":
    main()
