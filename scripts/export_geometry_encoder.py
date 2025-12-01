"""
Export SAM3 Geometry Encoder using torch.export.

The Geometry Encoder encodes geometric prompts (points, boxes, masks) into embeddings.
For standalone export, we use pre-computed vision features and create a simplified interface.
"""

import torch
import torch.nn as nn
import mock_setup
from sam3.model_builder import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt

class GeometryEncoderWrapper(nn.Module):
    """
    Wrapper for standalone geometry encoder export.
    
    Uses pre-computed vision features (from one forward pass of vision backbone)
    and provides a simplified tensor-based interface instead of Prompt objects.
    """
    def __init__(self, geometry_encoder, vision_features, img_sizes, vision_pos_enc):
        super().__init__()
        self.geometry_encoder = geometry_encoder
        
        # Store vision features as buffers (part of module state)
        for i, feat in enumerate(vision_features):
            self.register_buffer(f'vision_feat_{i}', feat)
        
        for i, pos in enumerate(vision_pos_enc):
            self.register_buffer(f'vision_pos_{i}', pos)
        
        self.img_sizes = img_sizes  # List of (H, W) tuples
        
    def forward(
        self,
        # Point inputs (sequence-first: [num_points, batch, ...])
        point_coords,      # [num_points, batch, 2] - normalized xy coordinates
        point_labels,      # [num_points, batch] - 1=foreground, 0=background
        point_mask,        # [batch, num_points] - False=valid
        # Box inputs (sequence-first: [num_boxes, batch, ...])
        box_coords,        # [num_boxes, batch, 4] - normalized cxcywh
        box_labels,        # [num_boxes, batch] - 1=positive
        box_mask,          # [batch, num_boxes] - False=valid
    ):
        """
        Forward pass with simplified tensor interface.
        
        All inputs use standard PyTorch tensor formats for easier export.
        """
        # Reconstruct Prompt object from tensors
        prompt = Prompt(
            point_embeddings=point_coords,
            point_labels=point_labels,
            point_mask=point_mask,
            box_embeddings=box_coords,
            box_labels=box_labels,
            box_mask=box_mask,
        )
        
        # Reconstruct vision feature lists from buffers
        vision_feats = [getattr(self, f'vision_feat_{i}') for i in range(4)]
        vision_pos = [getattr(self, f'vision_pos_{i}') for i in range(4)]
        
        # Call geometry encoder
        geo_embeds, geo_mask = self.geometry_encoder(
            geo_prompt=prompt,
            img_feats=vision_feats,
            img_sizes=self.img_sizes,
            img_pos_embeds=vision_pos,
        )
        
        # Return just the embeddings (mask is hard to export)
        # In production, you'd need the mask too, but for testing this is fine
        return geo_embeds

def export_geometry_encoder():
    """Export the geometry encoder with pre-computed vision features."""
    
    print("=== Building SAM3 Model ===")
    model = build_sam3_image_model(
        checkpoint_path=None,
        eval_mode=True,
        load_from_HF=False,
    )
    
    geometry_encoder = model.geometry_encoder
    vision_backbone = model.backbone.vision_backbone
    
    # Force CPU to avoid device mismatch issues
    geometry_encoder = geometry_encoder.to('cpu')
    vision_backbone = vision_backbone.to('cpu')
    
    geometry_encoder.eval()
    vision_backbone.eval()
    
    # Get vision features from a sample image
    print("\n=== Computing Vision Features ===")
    dummy_image = torch.randn(2, 3, 1008, 1008)  # Batch of 2
    normalized_image = (dummy_image - 0.5) / 0.5
    
    with torch.no_grad():
        sam3_features, sam3_pos_enc, _, _ = vision_backbone(normalized_image)
    
    print(f"Vision features: {len(sam3_features)} scales")
    for i, feat in enumerate(sam3_features):
        print(f"  Scale {i}: {feat.shape}")
    
    # These features are [B, C, H, W], need to convert to sequence-first for geo encoder
    # But wait - let me check the actual format expected...
    # Actually, looking at the neck code, it returns [B, C, H, W]
    # But geometry_encoder expects sequence-first: [H*W, B, C]
    
    # Convert to sequence-first format
    print("\n=== Converting to Sequence-First Format ===")
    seq_first_features = []
    img_sizes = []
    
    for feat in sam3_features:
        B, C, H, W = feat.shape
        # Reshape to [H*W, B, C]
        feat_seq = feat.permute(2, 3, 0, 1).reshape(H*W, B, C)
        seq_first_features.append(feat_seq)
        img_sizes.append((H, W))
        print(f"  Converted {feat.shape} -> {feat_seq.shape}, img_size={img_sizes[-1]}")
    
    # Do the same for position encodings
    seq_first_pos = []
    for pos in sam3_pos_enc:
        B, C, H, W = pos.shape
        pos_seq = pos.permute(2, 3, 0, 1).reshape(H*W, B, C)
        seq_first_pos.append(pos_seq)
    
    # Create dummy prompts (sequence-first format)
    print("\n=== Creating Dummy Prompts ===")
    batch_size = 2
    
    # Points: [num_points, batch, 2]
    point_coords = torch.tensor([
        [[0.5, 0.5], [0.3, 0.7]],  # First point for each batch
        [[0.6, 0.4], [0.8, 0.2]],  # Second point
    ], dtype=torch.float32)
    
    point_labels = torch.tensor([
        [1, 1],  # Both foreground
        [1, 0],  # Mixed
    ], dtype=torch.long)
    
    point_mask = torch.tensor([
        [False, False],  # Batch 0: 2 valid points
        [False, False],  # Batch 1: 2 valid points
    ], dtype=torch.bool)
    
    # Boxes: [num_boxes, batch, 4]
    box_coords = torch.tensor([
        [[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]],
    ], dtype=torch.float32)
    
    box_labels = torch.tensor([
        [1, 1],
    ], dtype=torch.long)
    
    box_mask = torch.tensor([
        [False],  # Batch 0: 1 valid box
        [False],  # Batch 1: 1 valid box
    ], dtype=torch.bool)
    
    print(f"Point coords: {point_coords.shape}")
    print(f"Box coords: {box_coords.shape}")
    
    # Create wrapper
    print("\n=== Creating Wrapper ===")
    wrapped_encoder = GeometryEncoderWrapper(
        geometry_encoder,
        seq_first_features,
        img_sizes,
        seq_first_pos,
    )
    wrapped_encoder.eval()
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    try:
        with torch.no_grad():
            geo_embeds = wrapped_encoder(
                point_coords,
                point_labels,
                point_mask,
                box_coords,
                box_labels,
                box_mask,
            )
        print(f"✓ Forward pass successful!")
        print(f"  Geometry embeddings: {geo_embeds.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Export
    print("\n=== Exporting with torch.export ===")
    try:
        exported_program = torch.export.export(
            wrapped_encoder,
            (
                point_coords,
                point_labels,
                point_mask,
                box_coords,
                box_labels,
                box_mask,
            ),
            strict=False,
        )
        print("✓ Successfully exported!")
        
        # Save
        torch.export.save(exported_program, "sam3_geometry_encoder_exported.pt2")
        print("  Saved to sam3_geometry_encoder_exported.pt2")
        
        return exported_program
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def import_to_tvm(exported_program):
    """Attempt to import to TVM."""
    
    if exported_program is None:
        print("No exported program to import")
        return
    
    print("\n=== Importing to TVM Relax ===")
    
    try:
        import tvm
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program
        
        print("Importing...")
        mod = from_exported_program(
            exported_program,
            keep_params_as_input=False
        )
        
        print("✓ Successfully imported to TVM Relax!")
        print("\n=== TVM Module Summary ===")
        print(mod)
        
        # Save
        with open("sam3_geometry_encoder_tvm.txt", "w") as f:
            f.write(str(mod))
        print("\n  Saved TVM IR to sam3_geometry_encoder_tvm.txt")
        
        return mod
        
    except Exception as e:
        print(f"✗ TVM import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    exported_program = export_geometry_encoder()
    if exported_program:
        tvm_mod = import_to_tvm(exported_program)
