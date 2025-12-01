import sys
import os
import torch

# Add external/sam3 to path
sys.path.append(os.path.abspath("external/sam3"))

# Mock triton
from unittest.mock import MagicMock
sys.modules["triton"] = MagicMock()
sys.modules["triton.language"] = MagicMock()

# Mock decord
sys.modules["decord"] = MagicMock()

from sam3.model_builder import build_sam3_image_model

def explore():
    print("Building SAM3 Image Model...")
    try:
        model = build_sam3_image_model(
            checkpoint_path=None, # Don't load weights for structure check
            load_from_HF=False,
            device="cpu",
            eval_mode=True
        )
        print("Model built successfully.")
        print(model)
        
        # Check for specific ops
        print("\nChecking for specific ops...")
        for name, module in model.named_modules():
            if "RoPE" in str(type(module)):
                print(f"Found RoPE: {name} -> {type(module)}")
            if "Attention" in str(type(module)):
                print(f"Found Attention: {name} -> {type(module)}")
            if "GridSample" in str(type(module)) or "grid_sample" in str(module):
                 print(f"Found GridSample: {name} -> {type(module)}")

    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore()
