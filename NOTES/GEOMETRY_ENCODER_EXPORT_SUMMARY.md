# Geometry Encoder Export - Summary

## Status: ✅ Exported | ❌ TVM Import Blocked

## What We Accomplished

### 1. Successfully Exported Geometry Encoder
- **File**: `sam3_geometry_encoder_exported.pt2`
- **Method**: `torch.export` with wrapper
- **Size**: TBD

### 2. Identified TVM Blockers

```
AssertionError: Unsupported function types ['roi_align.default', 'scatter.src']
```

**Critical Missing Ops**:
- `roi_align.default` - ROI (Region of Interest) Align operation
- `scatter.src` - Scatter operation for tensor indexing

## Technical Details

### Wrapper Strategy
Created `GeometryEncoderWrapper` that:
- Stores pre-computed vision features as buffers
- Provides simplified tensor interface (no Prompt objects)
- Handles sequence-first format conversion
- Returns only embeddings (not masks)

### Device Compatibility Patch
**File**: `external/sam3/sam3/model/geometry_encoders.py:659`
```python
# Before (broken on CPU):
scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)

# After (works on CPU):
scale = scale.to(device=boxes_xyxy.device)
```

## TVM Operation Analysis

### `roi_align` (Region of Interest Align)
- **Purpose**: Extract features from bounding box regions
- **Usage**: `torchvision.ops.roi_align(img_feats, boxes_xyxy, roi_size)`
- **TVM Status**: ❌ Not supported in PyTorch frontend
- **Alternatives**:
  1. Check if TVM has vision.roi_align in C++ ops
  2. Implement custom TVM operator
  3. Replace with grid_sample + interpolation

### `scatter` 
- **Purpose**: Scatter/gather operations for tensor indexing
- **Usage**: Padding mask manipulation, sequence assembly
- **TVM Status**: ❌ Not supported in PyTorch frontend
- **Alternatives**:
  1. Check TVM's scatter implementation
  2. Replace with equivalent ops
  3. Custom operator

## Next Steps

### Option 1: Implement Missing Ops in TVM
**Pros:**
- Complete solution
- Benefits all future models

**Cons:**
- Requires TVM C++/Python development
- Time-consuming

### Option 2: Replace with Supported Ops
**Pros:**
- Faster workaround
- Works with current TVM

**Cons:**
- May have performance impact
- Requires SAM3 code modification

### Option 3: Partial Export
**Pros:**
- Export what works
- Isolate unsupported parts

**Cons:**
- Incomplete pipeline
- Complex data flow management

## Recommendation

Given the blockers:
1. **Short-term**: Document op requirements, try other components
2. **Medium-term**: Research TVM's vision ops library for roi_align
3. **Long-term**: Contribute missing ops to TVM or wait for support

The good news: We successfully **exported** the geometry encoder, proving the approach works. The TVM import blocker is a known limitation, not a fundamental issue.

## Files

- **Export Script**: `scripts/export_geometry_encoder.py`
- **Exported Model**: `sam3_geometry_encoder_exported.pt2`
- **Source Patch**: `external/sam3/sam3/model/geometry_encoders.py:659`
- **This Summary**: `NOTES/GEOMETRY_ENCODER_EXPORT_SUMMARY.md`
- **Analysis**: `NOTES/GEOMETRY_ENCODER_ANALYSIS.md`
