# TVM Import Attempt - Summary

## Status: Blocked

We successfully exported the SAM3 vision backbone using `torch.export`, but encountered a blocker when importing to TVM Relax.

## What Worked

1. ✅ **Model building** with correct 1008×1008 input size
2. ✅ **torch.export** - Successfully exported to ExportedProgram format
3. ✅ **Saved exported model** to `sam3_vision_backbone_exported.pt2`

## Blocker: Complex Number Support

### Error
```
NotImplementedError: input_type torch.complex64 is not handled yet
```

### Root Cause
The RoPE (Rotary Position Encoding) implementation in SAM3 uses **complex numbers**:
- File: `external/sam3/sam3/model/vitdet.py`
- Functions: `compute_axial_cis()`, `apply_rotary_enc()`
- Uses: `torch.polar()`, `torch.view_as_complex()`, complex multiplication

### Why This Matters
```python
# From vitdet.py
freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)  # complex64
freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)  # complex64
freqs_cis = torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)     # complex64

# Later in apply_rotary_enc:
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # Complex multiplication
```

TVM's PyTorch frontend doesn't support `torch.complex64` datatype yet.

## Potential Solutions

### Option 1: Decompose Complex Operations (Recommended)
Replace complex number operations with equivalent real number operations:
- `complex_mul(a, b)` → Manual real/imag calculations
- Modify `apply_rotary_enc()` to work with real numbers only
- This makes the model TVM-compatible

### Option 2: Use TorchScript as Intermediate
- Keep the TorchScript model (`sam3_vision_backbone_traced.pt`)
- Write custom TVM operators for the complex ops
- More work, but preserves exact behavior

### Option 3: Wait for TVM Support
- File an issue/PR with TVM to add `torch.complex64` support
- Not viable for immediate porting

### Option 4: Use a different backend
- Try ONNX export (but ONNX also has limited complex number support)
- Try TensorRT (supports more PyTorch ops natively)

## Recommendation

**Short-term**: Implement Option 1 - decompose RoPE's complex operations into real operations.

This involves:
1. Fork/patch `external/sam3/sam3/model/vitdet.py`
2. Rewrite `apply_rotary_enc()` to avoid complex numbers:
   ```python
   # Instead of:
   xq_ = torch.view_as_complex(xq)
   xq_out = torch.view_as_real(xq_ * freqs_cis)
   
   # Do:
   # Split into real and imaginary parts
   # Manually compute complex multiplication
   # a*b = (a_real*b_real - a_imag*b_imag) + i(a_real*b_imag + a_imag*b_real)
   ```
3. Re-export and import to TVM

**Long-term**: Contribute complex number support to TVM's PyTorch frontend.

## Next Steps

1. Create a branch in sam3 submodule with RoPE decomposition
2. Test that modified model produces same outputs
3. Re-export and import to TVM
4. Continue with TVM optimization

## Files Created

- `sam3_vision_backbone_exported.pt2` - Exported program (but can't import to TVM yet)
- `scripts/export_and_import_tvm.py` - Export and import script
- This summary document
