"""
Export SAM3 vision backbone using torch.export and import to TVM Relax.

This is the recommended approach for PyTorch 2.0+ models.
"""

import torch
import mock_setup
from sam3.model_builder import build_sam3_image_model

class VisionBackboneWrapper(torch.nn.Module):
    """Wrapper for vision backbone that returns only tensors."""
    def __init__(self, vision_backbone):
        super().__init__()
        self.backbone = vision_backbone
    
    def forward(self, x):
        sam3_out, sam3_pos, sam2_out, sam2_pos = self.backbone(x)
        # Return only SAM3 outputs as a flat tuple for easier handling
        # Flatten the lists into a single tuple
        return tuple(sam3_out + sam3_pos)

def export_with_dynamo():
    """Export vision backbone using torch.export."""
    
    print("=== Building SAM3 Model ===")
    model = build_sam3_image_model(
        checkpoint_path=None,
        eval_mode=True,
        load_from_HF=False,
    )
    
    wrapped_backbone = VisionBackboneWrapper(model.backbone.vision_backbone)
    wrapped_backbone.eval()
    
    # Prepare input
    print("\n=== Preparing Input ===")
    dummy_input = torch.randn(1, 3, 1008, 1008)
    normalized_input = (dummy_input - 0.5) / 0.5
    print(f"Input shape: {normalized_input.shape}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        output = wrapped_backbone(normalized_input)
    print(f"✓ Forward pass successful")
    print(f"  Output: tuple of {len(output)} tensors")
    for i, t in enumerate(output):
        print(f"    [{i}]: {t.shape}")
    
    # Export using torch.export
    print("\n=== Exporting with torch.export ===")
    try:
        exported_program = torch.export.export(
            wrapped_backbone,
            (normalized_input,),
            strict=False  # Allow some flexibility
        )
        print("✓ Successfully exported with torch.export!")
        
        # Save the exported program
        torch.export.save(exported_program, "sam3_vision_backbone_exported.pt2")
        print("  Saved to sam3_vision_backbone_exported.pt2")
        
        return exported_program
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def import_to_tvm(exported_program):
    """Import the exported program to TVM."""
    
    if exported_program is None:
        print("No exported program to import")
        return
    
    print("\n=== Importing to TVM Relax ===")
    
    try:
        import tvm
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program
        
        # Prepare input spec
        input_shape = (1, 3, 1008, 1008)
        
        print(f"Input shape: {input_shape}")
        print("Importing...")
        
        # Import from exported program
        mod = from_exported_program(
            exported_program,
            keep_params_as_input=False
        )
        
        print("✓ Successfully imported to TVM Relax!")
        print("\n=== TVM Module Summary ===")
        print(mod)
        
        # Save TVM module
        print("\n=== Saving TVM Module ===")
        with open("sam3_vision_backbone_tvm.txt", "w") as f:
            f.write(str(mod))
        print("  Saved TVM IR to sam3_vision_backbone_tvm.txt")
        
        return mod
        
    except Exception as e:
        print(f"✗ TVM import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    exported_program = export_with_dynamo()
    if exported_program:
        tvm_mod = import_to_tvm(exported_program)
