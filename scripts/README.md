# SAM3 TVM Port - Scripts

This directory contains the working scripts for porting SAM3 to TVM.

## Files

### `mock_setup.py`
Mocks dependencies (triton, decord) that aren't available on macOS.
Must be imported before SAM3 modules.

Usage:
```python
import mock_setup  # Must be first
from sam3.model_builder import build_sam3_image_model
```

### `export_vision_backbone.py`
**Main script** - Exports the vision backbone using torch.export and attempts TVM import.

**What it does:**
- Builds SAM3 model
- Wraps vision backbone to flatten outputs into a tuple
- Uses correct 1008×1008 input size
- Applies normalization (mean/std = 0.5)
- Exports using torch.export (PyTorch Dynamo)
- Saves to `sam3_vision_backbone_exported.pt2`
- Attempts TVM import via `tvm.relax.frontend.torch.from_exported_program()`

**Output:** `sam3_vision_backbone_exported.pt2` (1.8GB)

**Status:** ⚠️ Export works, TVM import blocked by torch.complex64 support (RoPE)

### `export_transformer_encoder.py`
Exports `TransformerEncoderFusion` to validate cross-attention + MHA support and tries TVM import.

**What it does:**
- Builds SAM3 and grabs the transformer encoder only
- Uses dummy tensors shaped like real inputs (vision features + text prompts)
- Flattens dict outputs to a tuple for `torch.export`
- Saves `sam3_transformer_encoder_exported.pt2` and optional TVM IR dump
- Loads `tvm_custom_ops.py` shim to inject missing `aten::prod.dim_int` converter (temporary hack)

**Status:** Draft/test harness for TVM op coverage

## Usage

```bash
py -3.13 scripts/export_vision_backbone.py
# or
py -3.13 scripts/export_transformer_encoder.py
```

## Next Steps

See `../NOTES/TVM_IMPORT_BLOCKER.md` for the complex number blocker and potential solutions.
