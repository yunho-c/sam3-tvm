import torch
import torch.nn as nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
import tvm

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Simulate RoPE: view as complex, rotate, view back
        # x: (B, C, H, W)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # Rotate by some angle (dummy)
        freqs = torch.randn_like(x_complex)
        x_rotated = x_complex * freqs
        return torch.view_as_real(x_rotated).flatten(-2)

def reproduce():
    print("=== Reproducing torch.complex64 failure ===")
    model = ComplexModel()
    
    # Dummy inputs
    # input: (1, 4, 16, 16) -> last dim must be even for complex view
    input = torch.randn(1, 4, 16, 16)
    
    # Export
    print("Exporting...")
    exported_program = export(model, (input,))
    
    # Import to TVM
    print("Importing to TVM...")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        print("✓ Success! (Unexpected)")
    except Exception as e:
        print(f"✓ Caught expected error: {e}")

if __name__ == "__main__":
    reproduce()
