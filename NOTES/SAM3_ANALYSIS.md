# SAM3 Analysis and TVM Porting Strategy

## Model Architecture Overview
SAM3 consists of three main components:
1.  **Image Encoder**: A ViT-based backbone (likely MAE or similar) that processes the input image into embeddings.
2.  **Prompt Encoder**: Encodes points, boxes, and masks into embeddings.
3.  **Mask Decoder**: A transformer-based decoder that predicts masks from image and prompt embeddings.

## Key Challenges & Dependencies

### 1. NVIDIA-Specific Dependencies
-   **Triton**: Used for optimized kernels. Not available on macOS/CPU.
-   **Flash Attention**: Used in `RoPEAttention`. Not available on macOS/CPU.
-   **Mitigation**:
    -   Mock `triton` and `flash_attn` for tracing.
    -   Replace `RoPEAttention` with a standard PyTorch implementation or a TVM-compatible equivalent during tracing/export.
    -   Use `torch.backends.cuda.sdp_kernel` context manager to force math or mem-efficient attention if needed.

### 2. Custom Operations
-   **RoPE (Rotary Positional Embeddings)**: Implemented in `sam3.sam.rope`. Needs to be ensured that it exports correctly to TVM.
-   **GridSample**: Used in `geometry_encoders.py`. TVM supports `grid_sample`, but need to verify opset compatibility.
-   **Deformable Attention**: If used (common in detection models), might require custom TVM ops. (Need to verify if SAM3 uses it).

### 3. Hardcoded CUDA Devices
-   Found hardcoded `device="cuda"` in `position_encoding.py` and `decoder.py`.
-   **Action**: Patched these files to support CPU/MPS.

## Porting Strategy

### Phase 1: Preparation
-   [x] Environment setup with Python 3.13.
-   [x] Patch hardcoded CUDA devices.
-   [x] Mock `triton` and `decord`.

### Phase 2: Component-wise Tracing
-   **Image Encoder**:
    -   Input: Image tensor (1, 3, 1024, 1024).
    -   Output: Image embeddings.
    -   Action: Trace using `torch.export`.
-   **Prompt Encoder**:
    -   Input: Points, Boxes, Masks.
    -   Output: Sparse and Dense embeddings.
    -   Action: Trace using `torch.export`.
-   **Mask Decoder**:
    -   Input: Image embeddings, Prompt embeddings.
    -   Output: Masks, Scores.
    -   Action: Trace using `torch.export`.

### Phase 3: TVM Integration
-   Import exported graphs into TVM Relax.
-   Verify op support.
-   Implement missing ops (e.g., RoPE if not automatically handled).
-   Run verification against PyTorch outputs.

## Next Steps
1.  Create a script to trace the Image Encoder.
2.  Create a script to trace the Prompt Encoder.
3.  Create a script to trace the Mask Decoder.
