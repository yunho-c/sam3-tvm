# SAM3 Vision Backbone Tracing - Summary

## Problem Solved
Successfully traced the SAM3 vision backbone using torch.jit.trace after identifying and fixing the root cause of RoPE assertion errors.

## Root Cause Analysis

### The Problem
```
AssertionError in external/sam3/sam3/model/vitdet.py:63
assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
```

### Why It Happened
1. **Wrong Input Size**: We were using 1024×1024, a common image size for segmentation models
2. **SAM3's Actual Requirement**: 1008×1008 (found in official examples)
3. **The Math**:
   - With 1024×1024 input and patch_size=16 → 64×64 feature maps
   - With 1008×1008 input and patch_size=16 → 63×63 feature maps
   - RoPE freqs_cis was precomputed for 63×63
   - Shape mismatch: `freqs_cis(63,63) != features(64,64)` → AssertionError

## Solution

### Key Findings from Official Examples
From `examples/sam3_image_batched_inference.py`:
```python
transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
```

### Implementation
1. **Correct Input Size**: 1008×1008 (not 1024×1024)
2. **Preprocessing**: Normalize with mean/std = [0.5, 0.5, 0.5]
3. **Output Handling**: Wrapped backbone to return only tensor outputs (no None values for TorchScript)

## Files Created

### `scripts/export_vision_backbone.py`
Main export script that:
- Builds the SAM3 model
- Wraps vision backbone to flatten outputs
- Uses correct 1008×1008 input size
- Applies proper normalization
- **Exports with torch.export** (PyTorch Dynamo)
- Attempts TVM import (blocked by complex64)
- Saves to `sam3_vision_backbone_exported.pt2` (1.8GB)

## Traced Model Details

**File**: `sam3_vision_backbone_exported.pt2`  
**Size**: 1.8GB  
**Format**: torch.export.ExportedProgram
**Input**: `[1, 3, 1008, 1008]` (normalized, float32)  
**Output**: Tuple of 8 tensors (4 feature scales + 4 position encodings)

**Feature Map Shapes** (example):
```
Scale 0: [1, 256, 252,  252]  # 4x upsampled
Scale 1: [1, 256, 126, 126]  # 2x upsampled  
Scale 2: [1, 256, 63, 63]    # 1x (base)
Scale 3: [1, 256, 31, 31]    # 0.5x downsampled
```

## Triton Mocking Strategy

### Current Approach (macOS)
Mocking triton to bypass import errors is **acceptable** for model exploration because:
- The RoPE error was unrelated to triton
- We're just inspecting model structure
- Flash attention fallback works on CPU

### For Production TVM Port
Recommendation:
- Use a CUDA machine for validation
- Disable triton-specific optimizations:
  ```python
  torch.backends.cuda.enable_flash_sdp(False)
  ```
- TVM will replace triton kernels with its own optimized implementations anyway

## Next Steps

For TVM porting:
1. ✅ Vision backbone traced with TorchScript
2. ⏭️ Try `torch.export` (Dynamo) for better TVM compatibility
3. ⏭️ Import traced model into TVM using `tvm.relax.frontend.torch`
4. ⏭️ Handle any missing operator conversions
5. ⏭️ Optimize with TVM's MetaSchedule

## Lessons Learned

1. **Always check official examples** - they contain critical implementation details
2. **Input size matters** - even small differences (1024 vs 1008) can break models with precomputed embeddings
3. **RoPE is sensitive to spatial dimensions** - freqs_cis must match feature map size exactly
4. **TorchScript doesn't like None** - wrap outputs to return only tensors
