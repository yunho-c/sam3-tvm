import torch
import torch.nn as nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
import tvm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import tvm_custom_ops # Apply patches

class ScatterModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, dim, index, src):
        # Replicates aten::scatter.src
        # input.scatter_(dim, index, src) or out-of-place scatter
        return torch.scatter(input, dim, index, src)

def reproduce():
    print("=== Reproducing aten::scatter.src failure ===")
    model = ScatterModel()
    
    # Dummy inputs
    # input: (3, 5)
    input = torch.zeros(3, 5, dtype=torch.float32)
    dim = 0
    # src: (2, 5)
    src = torch.ones(2, 5, dtype=torch.float32)
    # index: (2, 5) - must match src shape
    index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64)
    
    # Export
    print("Exporting...")
    exported_program = export(model, (input, dim, index, src))
    
    # Import to TVM
    print("Importing to TVM...")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ Success! (Unexpected)")
    except Exception as e:
        print(f"✓ Caught expected error: {e}")

if __name__ == "__main__":
    reproduce()
