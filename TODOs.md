# SAM3 to TVM Port - TODOs

This document tracks the progress of porting SAM3 to TVM.


 ## SAM3 Architecture Overview
 
 ```
 ┌─────────────────────────────────────────────────────────┐
 │ Input: Image (1008×1008) + Prompts (text/box/point)     │
 └─────────────────────────────────────────────────────────┘
                           ↓
 ┌─────────────────────────────────────────────────────────┐
 │ 1. BACKBONE                                             │
 │    ├─ Vision Encoder (ViT + RoPE)                       │
 │    ├─ Text Encoder (CLIP-like)                          │
 │    └─ VL Combiner (fusion)                              │
 └─────────────────────────────────────────────────────────┘
                           ↓
 ┌─────────────────────────────────────────────────────────┐
 │ 2. GEOMETRY ENCODER                                     │
 │    ├─ Point encoding                                    │
 │    ├─ Box encoding (roi_align)                          │
 │    └─ Mask encoding (grid_sample)                       │
 └─────────────────────────────────────────────────────────┘
                           ↓
 ┌─────────────────────────────────────────────────────────┐
 │ 3. TRANSFORMER ENCODER                                  │
 │    ├─ Cross-attention (vision + prompts)                │
 │    └─ Multi-scale fusion                                │
 └─────────────────────────────────────────────────────────┘
                           ↓
 ┌─────────────────────────────────────────────────────────┐
 │ 4. TRANSFORMER DECODER                                  │
 │    ├─ Object queries                                    │
 │    ├─ Self-attention                                    │
 │    ├─ Cross-attention                                   │
 │    ├─ RoPE attention (complex64 issue!)                 │
 │    └─ Box refinement                                    │
 └─────────────────────────────────────────────────────────┘
                           ↓
 ┌─────────────────────────────────────────────────────────┐
 │ 5. OUTPUT HEADS                                         │
 │    ├─ Segmentation Head (pixel-wise masks)              │
 │    └─ Scoring Head (confidence scores)                  │
 └─────────────────────────────────────────────────────────┘
 ```

 ## ✅ Phase 1: Component Export & Import
 
 #### [x] Export Vision Backbone (**Exported & Imported**)
 - **File**: `sam3/model/image_encoder.py` (via `build_sam3_image_model`)
 - **Script**: `scripts/export_vision_backbone.py`
 - **Export**: ✅ SUCCESS - saved to `sam3_vision_backbone_exported.pt2`
 - **TVM Import**: ✅ SUCCESS
 - **Blockers Resolved**:
   - `complex64` (RoPE) patched via `scripts/patch_rope.py`
 - **Notes**:
   - Uses `patch_rope.py` to replace complex RoPE with float arithmetic.
   - Output is a tuple of multi-scale feature maps.
 
 #### [x] Export Geometry Encoder (**Exported & Imported**)
 - **File**: `sam3/model/geometry_encoders.py::SequenceGeometryEncoder`
 - **Script**: `scripts/export_geometry_encoder.py`
 - **Export**: ✅ SUCCESS - saved to `sam3_geometry_encoder_exported.pt2`
 - **TVM Import**: ✅ SUCCESS
 - **Blockers Resolved**:
   - `scatter.src` (patched via `scripts/tvm_custom_ops.py`)
   - `roi_align` (patched via `scripts/tvm_custom_ops.py`)
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
 - **TVM Import**: ✅ SUCCESS - saved to `sam3_image_encoder_tvm.txt`
 - **Ops verified**: 
   - `floor_divide` (patched)
   - `sin`/`cos` (RoPE patched to avoid complex64)
   - `matmul`, `layer_norm`, `relu`
 
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
 
 ---
 
 ## Phase 2: TVM Import & Op Support
 
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
 
 #### [x] Import Vision Backbone to TVM Relax
 - **Status**: ✅ Imported (RoPE patched)
 - **Verification**: [ ] Compare outputs with PyTorch
 
 #### [x] Import Geometry Encoder to TVM Relax
 - **Status**: ✅ Imported (scatter/roi_align patched)
 - **Verification**: [ ] Compare outputs with PyTorch
 
 #### [x] Import Transformer Encoder to TVM Relax
 - **Status**: ✅ Imported
 - **Verification**: [ ] Compare outputs with PyTorch
 
 #### [x] Import Transformer Decoder to TVM Relax
 - **Status**: ✅ Imported
 - **Verification**: [ ] Compare outputs with PyTorch
 
 #### [x] Import Output Heads to TVM Relax
 - **Status**: ✅ Imported
 - **Verification**: [ ] Compare outputs with PyTorch
 
 #### [ ] End-to-End TVM Module
 - **Tasks**:
   - [ ] Combine all components
   - [ ] Handle multi-input/multi-output
   - [ ] Verify correctness vs PyTorch
   - [ ] Document any limitations
 
 ---
 
 ## Phase 3: Optimization & Verification
 
 ### Correctness Verification
 
 #### Verification Framework
 - [x] Create verification script: `scripts/verify_components.py`
 - [ ] Run component-level verification
   - [ ] Vision backbone
   - [ ] Geometry encoder
   - [ ] Transformer encoder
   - [ ] Transformer decoder
   - [ ] Segmentation head
   - [ ] Scoring head
 - [ ] End-to-end output comparison
 - [ ] Document results in `VERIFICATION_REPORT.md`
 
 **Methodology:**
 - Numerical tolerance: `rtol=1e-5, atol=1e-5`
 - Metrics tracked: max absolute error, mean absolute error, max relative error
 - Output: `verification_report.json` with detailed comparison results
 
 **Usage:**
 ```bash
 py -3.13 scripts/verify_components.py
 ```
 
 ### Performance Optimization
 
 #### Benchmarking
 - [ ] Create benchmarking script: `scripts/benchmark_components.py`
 - [ ] Establish baseline performance (PyTorch vs TVM unoptimized)
   - [ ] Latency (mean, median, p95, p99)
   - [ ] Throughput (inferences/sec)
   - [ ] Peak memory usage
 - [ ] Document baseline metrics
 
 #### TVM MetaSchedule Tuning
 - [ ] Create tuning script: `scripts/tune_with_metaschedule.py`
 - [ ] Apply auto-tuning to vision backbone
 - [ ] Apply auto-tuning to full model (optional)
 - [ ] Save tuned parameters for reuse
 - [ ] Compare performance before/after tuning
 
 **Performance Goals:**
 - TVM (unoptimized) within 2x of PyTorch
 - TVM (tuned) matches or exceeds PyTorch performance
 
 ### Deployment
 
 #### Multi-Target Compilation
 - [ ] Create compilation script: `scripts/compile_for_targets.py`
 - [ ] Compile for CPU (LLVM x86_64)
 - [ ] Compile for CUDA (if available)
 - [ ] Compile for Metal (Apple Silicon, if available)
 - [ ] Validate compilation success for each target
 - [ ] Save compiled artifacts
 
 #### Documentation
 - [ ] Create deployment guide: `DEPLOYMENT.md`
   - [ ] Compilation instructions for each target
   - [ ] Loading and running compiled models
   - [ ] Performance characteristics
   - [ ] Known limitations and workarounds
 - [ ] Create verification report: `VERIFICATION_REPORT.md`
   - [ ] Numerical accuracy results
   - [ ] Known discrepancies and causes
   - [ ] Tolerance thresholds
   - [ ] Production recommendations


---

## References

- [NOTES/SAM3_ANALYSIS.md](NOTES/SAM3_ANALYSIS.md) - Architecture analysis
- [NOTES/VISION_BACKBONE_TRACING_SUMMARY.md](NOTES/VISION_BACKBONE_TRACING_SUMMARY.md) - Vision backbone work
- [NOTES/TVM_IMPORT_BLOCKER.md](NOTES/TVM_IMPORT_BLOCKER.md) - Complex64 blocker details
- [scripts/README.md](scripts/README.md) - Scripts documentation
