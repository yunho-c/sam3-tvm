# TVM Integration Troubleshooting Lessons

This document summarizes the key issues encountered and lessons learned while integrating SAM3 with Apache TVM Relax.

## 1. Boolean Type Inference & "Safe Softmax"
**Issue:** `tvm.error.InternalError: Cannot decide min_value for typebool`
**Context:** Occurred during the import of Transformer components (Encoder, Decoder, Heads).
**Cause:** 
The error is triggered when `float("-inf")` is used in `masked_fill_` on a boolean mask (e.g., `attn_mask` or `key_padding_mask`). This pattern often invokes a "safe softmax" logic in lower-level TVM passes or optimizations, which attempts to compute the maximum value of the tensor to ensure numerical stability. Since the tensor is boolean (or derived from one in a way that preserves some boolean properties in the IR), the `min_value` or `max` operation fails because it's not defined for boolean types in this context.
**Fix:**
Replace `float("-inf")` with a large negative finite number, such as `-1e9`.
```python
# Before
new_mask.masked_fill_(attn_mask, float("-inf"))

# After
new_mask.masked_fill_(attn_mask, -1e9)
```
This avoids the specific code path that triggers the boolean min/max check.

## 2. Dynamic Shapes & `resize2d`
**Issue:** `resize2d` failures or constraint violations during export/import.
**Context:** Segmentation Head and Pixel Decoder using `F.interpolate`.
**Cause:**
TVM's `resize2d` operator, especially when derived from `F.interpolate`, can be problematic with fully dynamic shapes (`torch.export.Dim`). It often expects `PrimExpr` that can be statically analyzed or specific constant scales. Dynamic output sizes calculated from dynamic input shapes can lead to complex symbolic expressions that fail validation.
**Fix:**
*   **Static Spatial Shapes:** For components heavily relying on resizing (like the Segmentation Head), use static spatial dimensions (e.g., `256x256`) during export, while keeping batch size or other dimensions dynamic if possible.
*   **Scale Factor:** Prefer `scale_factor` over explicit `size` in `F.interpolate` if the framework supports it, as it simplifies the graph.

## 3. Type Mismatches in `floor_divide`
**Issue:** `TypeError: Binary operators must have the same datatype` (e.g., `float32` vs `int32`).
**Context:** Positional encoding or arithmetic operations involving division.
**Cause:**
PyTorch's `floor_divide` allows mixed types (e.g., dividing a float tensor by an int scalar). TVM's `floor_divide` (and many other binary ops) strictly requires operands to have the same data type.
**Fix:**
Monkeypatch `BaseFXGraphImporter._div` or implement a custom converter. Check the types of operands. If one is float and the other is int, cast the int operand to float and use `floor(divide(a, b))` instead of `floor_divide`.
```python
if "float" in dtype1 and "int" in dtype2:
    inp_2 = bb.emit(relax.op.astype(inp_2, dtype1))
    return bb.emit(relax.op.floor(relax.op.divide(inp_1, inp_2)))
```

## 4. `tir.SizeVar` in Assertions
**Issue:** `AssertionError: Unsupported function type` for `aten::eq` or `aten::_assert_scalar`.
**Context:** `torch.export` generates assertions for dynamic shape guards.
**Cause:**
The arguments to these assertions are often `tir.SizeVar` (symbolic integers representing shapes). TVM's default converters might not handle `tir.SizeVar` directly in `relax.op.equal`.
**Fix:**
Implement custom converters for `eq` and `_assert_scalar`. Explicitly wrap `tir.SizeVar` in `relax.PrimValue` and constant scalars in `relax.const` before passing them to Relax operators.

## 5. `aten::slice` Argument Handling
**Issue:** `IndexError` or `OpaquePyObject` errors.
**Context:** Slicing tensors with optional start/end/step or symbolic indices.
**Cause:**
`aten::slice` arguments can be `None`, single-element lists, or symbolic variables. `None` needs to be converted to default values (0, INT_MAX, 1). Lists need unwrapping. Symbolic variables require using `dynamic_strided_slice` instead of `strided_slice`.
**Fix:**
Implement a robust `_slice` converter with a `clean_arg` helper that handles:
*   `None` -> Defaults.
*   `[x]` -> `x`.
*   `tir.SizeVar` -> `relax.PrimValue`.
*   Switching between `strided_slice` (static) and `dynamic_strided_slice` (dynamic) based on argument types.

## 6. LayerNorm Decomposition
**Issue:** LLVM module verification failures (debug info related).
**Context:** Compiling models with `nn.LayerNorm`.
**Cause:**
In some TVM/LLVM environment configurations, the native `relax.nn.layer_norm` implementation triggers an issue during code generation.
**Fix:**
Use `relax.transform.LegalizeOps` to decompose `relax.nn.layer_norm` into its constituent operations (`mean`, `sub`, `mul`, `sqrt`, `div`, `add`).

## 7. `grid_sample` & Rank Requirements
**Issue:** Rank mismatch errors in subsequent operations.
**Context:** Geometry Encoder using `grid_sample` (or a bypass).
**Cause:**
`grid_sample` expects 4D input `(B, C, H, W)`. If the output is squeezed to 3D `(B, C, N)` but subsequent operations (like `permute`) expect or imply a different rank, it causes failures.
**Fix:**
Ensure explicit `view` or `reshape` operations are used to maintain the expected rank, especially when bypassing ops or dealing with operations that might drop dimensions.

## 8. Consistent Patching with Shared Modules
**Issue:** Inconsistent application of patches (e.g., `ManualMultiheadAttention`) across multiple export scripts running in the same process.
**Context:** Integration tests importing multiple export modules where some patch `torch.nn.MultiheadAttention` and others define local versions.
**Cause:**
Python's module caching and global state modifications can lead to race conditions or order-dependent behavior. If one module patches a global class and another imports the original before the patch, or defines a local version but uses the global one implicitly, behavior becomes unpredictable.
**Fix:**
Consolidate the patched class (e.g., `ManualMultiheadAttention`) into a single shared module (e.g., `scripts/manual_attention.py`). Import this shared class in all export scripts and apply the global patch consistently at the top level. This ensures all components use the exact same implementation with necessary fixes (like the `-1e9` mask fix).
