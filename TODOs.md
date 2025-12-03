# SAM3 to TVM Port - TODOs

This document tracks the progress of porting SAM3 to TVM.

## Current Status

### ✅ Phase 1: Vision Backbone Export (COMPLETE)
- [x] Diagnosed RoPE shape mismatch (1024 vs 1008 input size)
- [x] Identified correct preprocessing (1008×1008, normalize with mean/std=0.5)
- [x] Exported vision backbone using `torch.export`
- [x] Saved to `sam3_vision_backbone_exported.pt2` (1.8GB)
- [x] Identified blocker: TVM doesn't support `torch.complex64` (used in RoPE)

**Key Insight**: TVM has RoPE implementations for LLMs (e.g., `tvm.relax.frontend.nn.llm.kv_cache`), which may help solve the complex64 issue.

---

## SAM3 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Input: Image (1008×1008) + Prompts (text/box/point)   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 1. BACKBONE                                      [DONE] │
│    ├─ Vision Encoder (ViT + RoPE)                      │
│    ├─ Text Encoder (CLIP-like)                         │
│    └─ VL Combiner (fusion)                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. GEOMETRY ENCODER                              [TODO] │
│    ├─ Point encoding                                    │
│    ├─ Box encoding (roi_align)                         │
│    └─ Mask encoding (grid_sample)                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. TRANSFORMER ENCODER                           [TODO] │
│    ├─ Cross-attention (vision + prompts)               │
│    └─ Multi-scale fusion                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. TRANSFORMER DECODER                           [TODO] │
│    ├─ Object queries                                    │
│    ├─ Self-attention                                    │
│    ├─ Cross-attention                                   │
│    ├─ RoPE attention (complex64 issue!)                │
│    └─ Box refinement                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 5. OUTPUT HEADS                                  [TODO] │
│    ├─ Segmentation Head (pixel-wise masks)             │
│    └─ Scoring Head (confidence scores)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 2: Export Remaining Components

### Priority 1: Core Components

#### [x] Export Geometry Encoder (**Exported - TVM import blocked**)
- **File**: `sam3/model/geometry_encoders.py::SequenceGeometryEncoder`
- **Script**: `scripts/export_geometry_encoder.py`
- **Export**: ✅ SUCCESS - saved to `sam3_geometry_encoder_exported.pt2`
- **TVM Import**: ❌ BLOCKED
- **TVM Blockers Identified**:
  - `scatter.src` ❌ Not supported (used in `concat_padded_sequences`)
  - `roi_align` ❓ Likely unsupported (masked by scatter failure)
- **Notes**: 
  - `SequenceGeometryEncoder` uses `TransformerEncoderLayer` which required careful reconstruction.
  - `concat_padded_sequences` uses `scatter` for variable length sequence handling.

#### [x] Export Transformer Encoder
- **File**: `sam3/model/encoder.py::TransformerEncoderFusion`
- **Script**: `scripts/export_transformer_encoder.py`
- **Export**: ✅ SUCCESS - saved to `sam3_transformer_encoder_exported.pt2`
- **TVM Import**: ✅ SUCCESS - saved to `sam3_transformer_encoder_tvm.txt`
- **Notes**:
    - Requires passing `feat_sizes` to `forward` to avoid a bug in `sam3/model/encoder.py`.
    - Input `src` must be flattened `[L, B, C]` when `feat_sizes` is used.
    - Fully supported by TVM Relax (standard attention ops).

#### [x] Export Transformer Decoder
- **File**: `sam3/model/decoder.py::TransformerDecoder`
- **Script**: `scripts/export_decoder.py`
- **Export**: ✅ SUCCESS - saved to `sam3_transformer_decoder_exported.pt2`
- **TVM Import**: ✅ SUCCESS
#### [x] Export Image Encoder
- **File**: `sam3/model/vitdet.py::ViT` + `sam3/model/necks.py::Sam3DualViTDetNeck`
- **Script**: `scripts/export_image_encoder.py`
- **Export**: ✅ SUCCESS - saved to `sam3_image_encoder_exported.pt2`
- **TVM Import**: ❌ BLOCKED
- **Blocker**: `NotImplementedError: input_type torch.complex64 is not handled yet` (due to RoPE).
- **Notes**:
    - Requires `complex64` support in TVM Relax frontend.
    - `Sam3DualViTDetNeck` expects a Tensor input (despite type hint saying List).
- **Patches Applied**:
  - `scripts/export_decoder.py`: Used `strict=True` in `torch.export.export` to correctly handle guards without monkeypatching `is_dynamo_compiling`.
  - `scripts/export_decoder.py`: Monkeypatched `BaseFXGraphImporter._div` to handle `floor_divide` type mismatch (float vs int).
  - `scripts/tvm_custom_ops.py`: Implemented custom converter for `aten::scatter.src` mapping to `relax.op.scatter_elements`.
  - `scripts/tvm_custom_ops.py`: Implemented custom converter for `torchvision::roi_align` mapping to `topi.image.crop_and_resize`.
- **Notes**: Did NOT hit RoPE complex64 issue (likely uses different position embedding or compiled away).
- **Ops verified**: 
  - `floor_divide` (patched)
  - `sin`/`cos` (for position embedding)
  - `matmul`, `layer_norm`, `relu`

### Priority 2: Output Heads

#### [x] Export Segmentation Head
- **File**: `sam3/model/maskformer_segmentation.py::UniversalSegmentationHead`
- **Script**: `scripts/export_segmentation_head.py`
- **Export**: ✅ SUCCESS - saved to `sam3_segmentation_head_exported.pt2`
- **TVM Import**: ✅ SUCCESS
- **Ops to verify**: 
  - Upsampling
  - Convolutions
  - Multi-scale aggregation
- **Notes**: Mocked `triton` and `decord` to bypass import errors. Adjusted dummy input shapes for `einsum`.

#### [x] Export Scoring Head
- **File**: `sam3/model/model_misc.py::DotProductScoring`
- **Script**: `scripts/export_scoring_head.py`
- **Export**: ✅ SUCCESS - saved to `sam3_scoring_head_exported.pt2`
- **TVM Import**: ✅ SUCCESS
- **Ops to verify**: 
  - Dot product
  - Linear layers
- **Notes**: Mocked `triton` and `decord`.

### Priority 3: Integration

#### [ ] Export End-to-End Model
- **Script**: `scripts/export_full_model.py`
- **Notes**: 
  - May need custom wrapper to handle data structures
  - Complex due to multiple inputs/outputs
- **Expected complexity**: High

---

## Phase 3: TVM Import & Op Support

### RoPE Implementation Strategy

#### Step 1: Research TVM's LLM RoPE
- [ ] Study `tvm.relax.frontend.nn.llm.kv_cache`
- [ ] Check if complex64 is implicitly handled or decomposed
- [ ] Identify reusable patterns
- [ ] Document findings in `NOTES/TVM_ROPE_RESEARCH.md`

#### Step 2: Choose Implementation Approach
**Options:**
- **Option A**: Decompose complex ops in SAM3 source (fork/patch)
- **Option B**: Add complex64 support to TVM PyTorch frontend
- **Option C**: Use TVM's existing RoPE ops for LLMs
- **Option D**: Write custom TVM operator for 2D RoPE

**Decision**: TBD after research

#### Step 3: Other Op Support
- [ ] Verify `roi_align` support in TVM
- [ ] Verify `grid_sample` support in TVM
- [ ] Check if deformable attention is used
- [ ] Document op support status

### TVM Import Tasks

#### [ ] Import Vision Backbone to TVM Relax
- **Blocker**: RoPE complex64 issue
- **Target**: Successful Relax IR generation
- **Verification**: Compare outputs with PyTorch

#### [ ] Import Geometry Encoder to TVM Relax
- **Dependencies**: roi_align and grid_sample support
- **Verification**: Compare outputs with PyTorch

#### [ ] Import Transformer Encoder to TVM Relax
- **Dependencies**: Cross-attention support
- **Verification**: Compare outputs with PyTorch

#### [ ] Import Transformer Decoder to TVM Relax
- **Dependencies**: RoPE solution from vision backbone
- **Verification**: Compare outputs with PyTorch

#### [ ] Import Output Heads to TVM Relax
- **Expected**: Should be straightforward
- **Verification**: Compare outputs with PyTorch

#### [ ] End-to-End TVM Module
- **Tasks**:
  - Combine all components
  - Handle multi-input/multi-output
  - Verify correctness vs PyTorch
  - Document any limitations

---

## Phase 4: Optimization & Verification

### Correctness Verification
- [ ] Component-level output comparison (PyTorch vs TVM)
  - [ ] Vision backbone
  - [ ] Geometry encoder
  - [ ] Transformer encoder
  - [ ] Transformer decoder
  - [ ] Segmentation head
  - [ ] Scoring head
- [ ] End-to-end output comparison
- [ ] Numerical tolerance testing
- [ ] Document comparison methodology

### Performance Optimization
- [ ] Apply TVM tuning (MetaSchedule)
  - [ ] Vision backbone
  - [ ] Full model
- [ ] Memory optimization
  - [ ] Analyze peak memory usage
  - [ ] Optimize memory layout
- [ ] Batch size optimization
- [ ] Compare with PyTorch performance
  - [ ] Latency
  - [ ] Throughput
  - [ ] Memory usage

### Deployment
- [ ] Target-specific compilation
  - [ ] CPU (x86_64)
  - [ ] GPU (CUDA)
  - [ ] Metal (Apple Silicon)
- [ ] Runtime optimization
- [ ] Benchmarking on target hardware
- [ ] Document deployment guide

---

## Immediate Next Steps

1. **Export Geometry Encoder** (`scripts/export_geometry_encoder.py`)
   - Simplest component to start with
   - Will reveal grid_sample/roi_align support status
   
2. **Upstream/replace prod shim**
   - Add proper `aten::prod.dim_int` converter to TVM or refactor shim away
   - Keep shim as temporary unblocker for transformer encoder import
   
3. **Export Decoder + Heads** 
   - Will hit RoPE issue again
   - Use same wrapper pattern as vision backbone
   
4. **Research RoPE in TVM**
   - Study LLM implementations
   - Prototype solution
   - Apply to all RoPE components

---

## References

- [NOTES/SAM3_ANALYSIS.md](NOTES/SAM3_ANALYSIS.md) - Architecture analysis
- [NOTES/VISION_BACKBONE_TRACING_SUMMARY.md](NOTES/VISION_BACKBONE_TRACING_SUMMARY.md) - Vision backbone work
- [NOTES/TVM_IMPORT_BLOCKER.md](NOTES/TVM_IMPORT_BLOCKER.md) - Complex64 blocker details
- [scripts/README.md](scripts/README.md) - Scripts documentation
