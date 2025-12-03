# Boolean Parameter Type Mismatch Investigation

**Date**: December 3, 2025  
**Status**: Root cause identified, solution proposed  
**Related Files**: 
- `NOTES/TVM_DEBUGGING_LESSONS.md`
- `NOTES/TVM_COMPILATION_ERROR_STRATEGIES.md`
- `scripts/integration_test_debug.py`
- `scripts/integration_test_passcontext.py`

---

## Executive Summary

We encountered a `tvm.error.InternalError: Cannot decide min_value for typebool` error during SAM3 TVM integration. Through systematic debugging, we discovered:

1. **The error occurred during COMPILATION, not import** - all components imported successfully to TVM Relax
2. **ManualMultiheadAttention was solving the WRONG problem** - it was actually CAUSING the typebool error
3. **The real issue is a dtype mismatch** - TVM's compiler expects consistent types (float32 vs bool)
4. **The optimal fix is at the TVM import level** - patch `tvm_custom_ops.py` to handle boolean parameters

---

## Timeline of Investigation

### Phase 1: Initial Error (Misdiagnosis)
**Error Message:**
```
tvm.error.InternalError: Cannot decide min_value for typebool
```

**Initial Hypothesis (INCORRECT):**
- Thought the error was during PyTorch → TVM import
- Believed `float("-inf")` in attention masks was causing boolean type inference issues
- Created `ManualMultiheadAttention` to replace `-inf` with `-1e9`

**Actions Taken:**
- Created `scripts/manual_attention.py` with custom attention implementation
- Patched all export scripts to use `ManualMultiheadAttention`
- Updated `NOTES/TVM_DEBUGGING_LESSONS.md` with "Consistent Patching" lesson

**Result:** ❌ Error persisted in integration tests, even though individual components worked

---

### Phase 2: Isolation Testing (Key Insight)
**Tool:** `scripts/integration_test_debug.py`

**Discovery:**
```
Component                 Import     Compile   
------------------------------------------------------------
Vision Backbone           ✓ PASS     ✓ PASS    
Geometry Encoder          ✓ PASS     ✓ PASS    
Transformer Encoder       ✓ PASS     ✓ PASS    
Transformer Decoder       ✓ PASS     ✗ FAIL    
Segmentation Head         ✓ PASS     ✗ FAIL    
Scoring Head              ✓ PASS     ✗ FAIL    
```

**Key Insight:** 
- ✅ ALL components import successfully to TVM Relax
- ❌ 3 components fail during `relax.build()` compilation
- **The error is in the COMPILATION phase, not import!**

**Implication:** `ManualMultiheadAttention` was unnecessary for import

---

### Phase 3: Optimization Level Testing (Negative Result)
**Tool:** `scripts/integration_test_passcontext.py`

**Hypothesis:** Maybe the error is caused by specific optimization passes

**Test:** Compiled all components with `opt_level` 0, 1, 2, 3

**Results:**
```
Component                 opt=0    opt=1    opt=2    opt=3
------------------------------------------------------------
Vision Backbone           ✓ PASS   ✓ PASS   ✓ PASS   ✓ PASS
Geometry Encoder          ✓ PASS   ✓ PASS   ✓ PASS   ✓ PASS
Segmentation Head         ✓ PASS   ✓ PASS   ✓ PASS   ✓ PASS
Transformer Encoder       ✗ FAIL   ✗ FAIL   ✗ FAIL   ✗ FAIL
Transformer Decoder       ✗ FAIL   ✗ FAIL   ✗ FAIL   ✗ FAIL
Scoring Head              ✗ FAIL   ✗ FAIL   ✗ FAIL   ✗ FAIL
```

**Conclusion:** 
- ❌ Optimization level control (Strategy 1) does NOT solve the problem
- The error persists even at `opt_level=0` (minimal optimization)
- **The issue is in the IR itself, not optimization passes**

---

### Phase 4: The Breakthrough (Reversion Test)

**Action:** Reverted all `ManualMultiheadAttention` changes to test if it was needed

**CRITICAL DISCOVERY:**

#### Before Reversion (with ManualMultiheadAttention):
```
tvm.error.InternalError: Cannot decide min_value for typebool
```

#### After Reversion (without ManualMultiheadAttention):
```
Check failed: x->dtype == y->dtype (float32 vs. bool) : 
x and y must have the same dtype: float32 vs
```

**🎉 BREAKTHROUGH REALIZATION:**

1. **ManualMultiheadAttention was CAUSING the `typebool` error, not fixing it!**
2. **The original PyTorch `MultiheadAttention` works fine**
3. **The real error is a simple dtype mismatch (float32 vs bool)**
4. **This is a compile-time type check, not a runtime error**

---

## Root Cause Analysis

### What's Actually Happening

1. **PyTorch Export is Correct:**
   - Examination of `*_debug_ir.txt` files shows PyTorch export already includes:
     ```python
     lv13: R.Tensor((1, 5), dtype="float32") = R.astype(prompt_mask, dtype="float32")
     ```
   - Boolean masks ARE being cast to float32 in the IR

2. **The Problem is Function Parameters:**
   - Some functions have boolean tensor parameters in their signatures
   - Example from `scoring_head_debug_ir.txt`:
     ```python
     def main(
         hs: R.Tensor((6, 1, 10, 256), dtype="float32"),
         prompt: R.Tensor((5, 1, 256), dtype="float32"),
         prompt_mask: R.Tensor((1, 5), dtype="bool"),  # ← Boolean parameter!
         ...
     )
     ```

3. **TVM's Compilation Issue:**
   - TVM's compiler performs type checking during `relax.build()`
   - When it encounters operations mixing float32 and bool, it fails
   - Even though the IR has `R.astype`, the function signature declares bool
   - TVM's type system doesn't handle this gracefully

### Why ManualMultiheadAttention Made It Worse

The `ManualMultiheadAttention` implementation likely introduced additional boolean operations or changed the IR structure in a way that triggered the `min_value` type inference issue. By removing it, we revealed the simpler underlying dtype mismatch.

---

## Affected Components

### Components with Boolean Parameters:
1. **Transformer Encoder** - likely has attention masks
2. **Transformer Decoder** - has attention masks and key_padding_masks
3. **Scoring Head** - has `prompt_mask: R.Tensor((1, 5), dtype="bool")`

### Components WITHOUT Boolean Parameters:
1. **Vision Backbone** - ✅ Compiles successfully
2. **Geometry Encoder** - ✅ Compiles successfully  
3. **Segmentation Head** - ✅ Compiles successfully

---

## Solution Options Analysis

### Option A: Cast at PyTorch Export Level ❌ **NOT RECOMMENDED**
```python
# In export scripts, before torch.export
if mask.dtype == torch.bool:
    mask = mask.float()
```

**Pros:**
- Handles it before TVM sees it

**Cons:**
- **Hacky** - changes model's actual behavior
- Loses semantic type information
- Requires modifying every export script
- Doesn't address root cause

---

### Option B: TVM Pre-processing Pass ⚠️ **ACCEPTABLE**
```python
class BoolToFloatCaster(PyExprMutator):
    def visit_function_(self, func):
        # Cast boolean parameters to float32
        ...
```

**Pros:**
- Centralized fix
- Doesn't modify model code

**Cons:**
- Adds complexity to compilation pipeline
- Still a workaround, not a fix

---

### Option C: Fix at TVM Import Level ✅ **OPTIMAL**
```python
# In tvm_custom_ops.py
def patch_boolean_parameters():
    """Cast boolean function parameters to float32 during TVM import"""
    from tvm.relax.frontend.torch import from_exported_program
    
    original_import = from_exported_program
    
    def patched_import(exported_program, **kwargs):
        mod = original_import(exported_program, **kwargs)
        mod = cast_bool_params_to_float(mod)  # Post-process
        return mod
    
    tvm.relax.frontend.torch.from_exported_program = patched_import
```

**Pros:**
- ✅ **Addresses root cause** - TVM's PyTorch frontend should handle bool parameters
- ✅ **Least hacky** - legitimate frontend conversion issue
- ✅ **Upstream-able** - could contribute back to TVM
- ✅ **Minimal code changes** - just `tvm_custom_ops.py`
- ✅ **Centralized** - one fix for all components
- ✅ **Semantically correct** - TVM doesn't natively support bool in all contexts

**Cons:**
- Requires understanding TVM's import mechanism
- Might need to patch internal TVM structures

---

## Recommended Implementation

### Step 1: Add Boolean Parameter Converter to `tvm_custom_ops.py`

```python
def cast_bool_params_to_float(mod: IRModule) -> IRModule:
    """
    Cast boolean function parameters to float32.
    
    TVM's compiler doesn't handle boolean parameters well in all contexts,
    causing dtype mismatch errors during relax.build(). This function
    post-processes the imported IRModule to cast boolean parameters.
    """
    from tvm import relax
    from tvm.relax import PyExprMutator
    
    class BoolParamCaster(PyExprMutator):
        def visit_function_(self, func):
            # Check parameters for boolean types
            new_params = []
            param_casts = {}
            
            for param in func.params:
                if hasattr(param.struct_info, 'dtype'):
                    if str(param.struct_info.dtype) == 'bool':
                        # Create new parameter with float32 dtype
                        new_param = relax.Var(
                            param.name_hint,
                            relax.TensorStructInfo(
                                param.struct_info.shape,
                                "float32"
                            )
                        )
                        new_params.append(new_param)
                        param_casts[param] = new_param
                    else:
                        new_params.append(param)
                else:
                    new_params.append(param)
            
            # If no boolean params, return original
            if not param_casts:
                return func
            
            # Rebuild function with new parameters
            # ... (implementation details)
            
    caster = BoolParamCaster()
    return caster.visit_module(mod)
```

### Step 2: Apply in Import Functions

```python
# In each export script's import_to_tvm function
def import_to_tvm(exported_program):
    from tvm.relax.frontend.torch import from_exported_program
    
    # Import normally
    mod = from_exported_program(exported_program, keep_params_as_input=False)
    
    # Cast boolean parameters (if patch is applied)
    # This happens automatically if tvm_custom_ops patches from_exported_program
    
    return mod
```

---

## Testing Strategy

1. **Unit Test:** Create a simple test case with boolean parameters
2. **Component Test:** Test each failing component individually
3. **Integration Test:** Run full `integration_test.py`
4. **Verification:** Compare outputs with PyTorch

---

## Lessons Learned

### 1. Always Isolate Import vs Compilation
- Use separate test scripts for import and compilation phases
- Don't assume error location from stack traces alone

### 2. Test Hypotheses by Removal
- Sometimes the "fix" is the problem
- Revert changes systematically to identify root cause

### 3. Optimization Levels Can Reveal IR Issues
- Testing with `opt_level=0` helps distinguish IR problems from optimization problems
- If it fails at opt_level=0, the IR itself has issues

### 4. Read Error Messages Carefully
- `Cannot decide min_value for typebool` ≠ `dtype mismatch (float32 vs bool)`
- Different errors indicate different root causes

### 5. TVM's Type System is Strict
- Boolean types are not automatically promoted to numeric types
- Function signatures must match operation requirements

---

## References

- **TVM Relax Documentation**: https://tvm.apache.org/docs/reference/api/python/relax/
- **PyTorch Export**: https://pytorch.org/docs/stable/export.html
- **Related Issue**: TVM doesn't auto-cast boolean parameters (potential upstream bug)

---

## Next Steps

1. ✅ Document findings (this file)
2. ⏳ Implement boolean parameter casting in `tvm_custom_ops.py`
3. ⏳ Test with failing components
4. ⏳ Update `integration_test.py` to verify fix
5. ⏳ Consider contributing fix upstream to TVM

---

## Appendix: Debug Commands

### Inspect IR for Boolean Types
```bash
grep -n 'dtype="bool"' *_debug_ir.txt
```

### Find Boolean Operations
```bash
grep -n 'R\.equal\|R\.greater\|R\.less' *_debug_ir.txt
```

### Check Compilation Errors
```bash
grep -B 5 "dtype mismatch\|typebool" integration_test_passcontext.log
```

### Test Individual Component
```bash
python3.13 scripts/export_decoder.py
```
