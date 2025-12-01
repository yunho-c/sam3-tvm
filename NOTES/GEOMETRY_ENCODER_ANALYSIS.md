# Geometry Encoder Analysis Summary

## Overview

The SAM3 Geometry Encoder (`SequenceGeometryEncoder`) is responsible for encoding user prompts (points, boxes, masks) into embeddings that guide the transformer.

## Key Operations

### 1. Point Encoding
- `torch.nn.functional.grid_sample` - Pool image features at point locations
- Linear projections
- Positional encoding addition

### 2. Box Encoding  
- `torchvision.ops.roi_align` - Extract features from bounding box regions
- Coordinate conversion (CxCyWH → XYXY)
- Conv2d pooling  
- Linear projections

### 3. Mask Encoding
- Conv2d downsampling
- Position encoding fusion

### 4. Sequence Assembly
- `torch.cat` for concatenation
- Padding mask manipulation

### 5. Optional Transformer
- Cross-attention layers
- LayerNorm

## TVM Support Status

| Operation | Status | Notes |
|-----------|--------|-------|
| `grid_sample` | ❓ Unknown | **Need to verify** |
| `roi_align` | ❓ Unknown | **Need to verify** |
| Linear layers | ✅ Supported | Standard |
| Conv2d | ✅ Supported | Standard |
| LayerNorm | ✅ Supported | Standard | 
| torch.cat | ✅ Supported | Basic op |
| Cross-attention | ✅ Supported | Standard |

## Export Challenges

### Why Standalone Export is Impractical

1. **Dependencies on Vision Backbone**
   - Requires multi-scale image features as input
   - Features are in specific sequence-first format

2. **Complex Data Structures**
   - Uses custom `Prompt` class
   - Multiple optional fields (points, boxes, masks)
   - Padding masks for variable-length sequences

3. **Conditional Pathways**
   - Different code paths for points vs boxes
   - Optional mask encoding
   - Optional transformer encoding

4. **Unusual Tensor Format**
   - Sequence-first: `[seq_len, batch, ...]`
   - Most vision models use batch-first

## Recommendation

**Export as part of the full end-to-end model**, not standalone.

### Rationale
- Too tightly coupled with other components
- Complex wrapper needed for standalone export
- Better to verify TVM supports the key ops first
- Full model export will handle data flow naturally

## Next Steps

1. **Verify TVM Op Support**
   ```python
   # Check TVM documentation/source for:
   - tvm.relax.op.grid_sample (or PyTorch frontend converter)
   - tvm.relax.op.roi_align (or PyTorch frontend converter)
   ```

2. **If Supported**
   - Proceed to full model export
   - Let TVM's PyTorch frontend handle the complexity

3. **If Not Supported**
   - Option A: Implement custom TVM operators
   - Option B: Replace with TVM-compatible alternatives
   - Option C: Keep in PyTorch, only export other components

## Files

- **Source Code**: `external/sam3/sam3/model/geometry_encoders.py`
- **This Document**: `NOTES/GEOMETRY_ENCODER_ANALYSIS.md`
