import torch
import torch.nn as nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
import tvm
import torchvision
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import tvm_custom_ops # Apply patches

class RoiAlignModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, boxes):
        # input: (N, C, H, W)
        # boxes: (K, 5) or list of boxes
        # output_size: (output_height, output_width)
        return torchvision.ops.roi_align(input, boxes, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=-1)

def reproduce():
    print("=== Reproducing torchvision::roi_align failure ===")
    model = RoiAlignModel()
    
    # Dummy inputs
    # input: (1, 4, 32, 32)
    input = torch.randn(1, 4, 32, 32)
    # boxes: (K, 5) [batch_index, x1, y1, x2, y2]
    # 2 boxes
    boxes = torch.tensor([
        [0, 10, 10, 20, 20],
        [0, 5, 5, 15, 15]
    ], dtype=torch.float32)
    
    # Export
    print("Exporting...")
    exported_program = export(model, (input, boxes))
    
    # Import to TVM
    print("Importing to TVM...")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ Success! (Unexpected)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✓ Caught expected error: {e}")

if __name__ == "__main__":
    reproduce()
