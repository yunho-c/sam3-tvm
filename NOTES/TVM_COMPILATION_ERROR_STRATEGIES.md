# TVM Compilation Error Debugging Strategies

This document provides general strategies for debugging TVM compilation errors, particularly those involving type mismatches, boolean operations, and optimization passes.

> [!NOTE]
> For the specific SAM3 boolean parameter investigation, see [`BOOLEAN_PARAMETER_INVESTIGATION.md`](./BOOLEAN_PARAMETER_INVESTIGATION.md)

---

## General Debugging Workflow

### Step 1: Isolate the Error Phase

Determine whether the error occurs during **import** or **compilation**:

```python
# Test import separately
from tvm.relax.frontend.torch import from_exported_program
try:
    mod = from_exported_program(exported_program)
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")

# Test compilation separately  
from tvm import relax
try:
    ex = relax.build(mod, target="llvm")
    print("✓ Compilation successful")
except Exception as e:
    print(f"✗ Compilation failed: {e}")
```

**Why this matters:** Import errors require fixing the PyTorch export or TVM frontend, while compilation errors require fixing the IR or optimization passes.

---

## Strategy 1: Control Optimization Passes

Use `PassContext` to diagnose optimization-related issues.

### Test Different Optimization Levels

```python
from tvm import relax, transform

# Test with minimal optimization
with transform.PassContext(opt_level=0):
    ex = relax.build(mod, target="llvm")
```

**Optimization Levels:**
- `0`: Minimal optimization, fastest compilation
- `1`: Basic optimizations
- `2`: Standard optimizations (default)
- `3`: Aggressive optimizations

**Interpretation:**
- If it works at `opt_level=0` but fails at higher levels → optimization pass issue
- If it fails at `opt_level=0` → fundamental IR issue

### Disable Specific Passes

```python
# Disable passes one by one to identify the culprit
with transform.PassContext(disabled_pass=["relax.LegalizeOps"]):
    ex = relax.build(mod, target="llvm")

# Common passes to test:
# - relax.LegalizeOps
# - relax.FoldConstant
# - relax.SimplifyExpr
# - relax.EliminateCommonSubexpr
```

### Use Custom Pipeline

```python
# Use minimal transformation pipeline
ex = relax.build(
    mod, 
    target="llvm",
    relax_pipeline="zero",  # or "default", "static_shape_tuning"
    exec_mode="bytecode"
)
```

---

## Strategy 2: Inspect the IR

Examine the Relax IR to understand what operations are causing issues.

### Save and Inspect IR

```python
# Save IR to file
with open("debug_ir.txt", "w") as f:
    f.write(str(mod))

# Or use PrintIR pass
from tvm import transform
with transform.PassContext():
    mod = transform.PrintIR()(mod)
```

### Search for Problematic Patterns

```bash
# Find boolean types
grep -n 'dtype="bool"' debug_ir.txt

# Find comparison operations that produce booleans
grep -n 'R\.equal\|R\.greater\|R\.less\|R\.not_equal' debug_ir.txt

# Find operations that might mix types
grep -n 'R\.add\|R\.subtract\|R\.multiply' debug_ir.txt | grep -i bool
```

**Common problematic patterns:**
```python
# Pattern 1: Boolean result used in arithmetic
lv1 = R.equal(x, y)  # dtype="bool"
lv2 = R.max(lv1)     # May fail if TVM expects numeric type

# Pattern 2: Type mismatch in operations
mask = R.Tensor(..., dtype="bool")
result = R.add(scores, mask)  # float32 + bool → error
```

---

## Strategy 3: Add Type Casting Pass

Create a custom transformation pass to handle type issues.

### Example: Cast Boolean Tensors to Float

```python
from tvm import relax
from tvm.relax import PyExprMutator
from tvm.ir import IRModule

class BoolToFloatCaster(PyExprMutator):
    """Cast boolean tensors to float32 before operations."""
    
    def visit_call_(self, call):
        # Check if operation involves boolean types
        if self._involves_bool_type(call):
            new_args = []
            for arg in call.args:
                if self._is_bool_tensor(arg):
                    # Insert explicit cast
                    new_args.append(relax.op.astype(arg, "float32"))
                else:
                    new_args.append(arg)
            return relax.Call(call.op, new_args, call.attrs, call.sinfo)
        return super().visit_call_(call)
    
    def _is_bool_tensor(self, expr):
        if hasattr(expr, 'struct_info'):
            sinfo = expr.struct_info
            if hasattr(sinfo, 'dtype'):
                return str(sinfo.dtype) == "bool"
        return False
    
    def _involves_bool_type(self, call):
        return any(self._is_bool_tensor(arg) for arg in call.args)

# Apply the pass
def sanitize_bool_types(mod: IRModule) -> IRModule:
    caster = BoolToFloatCaster()
    return caster.visit_module(mod)

# Usage
mod = sanitize_bool_types(mod)
ex = relax.build(mod, target="llvm")
```

---

## Strategy 4: Fix at Import Level

For issues with PyTorch → TVM translation, patch the import process.

### Example: Patch in `tvm_custom_ops.py`

```python
# Add to tvm_custom_ops.py
def patch_import_for_bool_params():
    """
    Patch TVM's PyTorch importer to handle boolean parameters.
    Apply this before importing any models.
    """
    from tvm.relax.frontend.torch import from_exported_program
    
    original_import = from_exported_program
    
    def patched_import(exported_program, **kwargs):
        mod = original_import(exported_program, **kwargs)
        # Post-process to fix boolean parameters
        mod = cast_bool_params_to_float(mod)
        return mod
    
    # Monkey-patch
    import tvm.relax.frontend.torch
    tvm.relax.frontend.torch.from_exported_program = patched_import
```

**When to use this:**
- Issue is in PyTorch export → TVM import translation
- Problem affects multiple components
- Want a centralized fix

---

## Debugging Tools

### 1. Integration Test with Error Isolation

```python
def test_component_with_error_handling(name, export_func, import_func):
    """Test import and compilation separately."""
    print(f"Testing {name}...")
    
    # Phase 1: Export
    try:
        prog = export_func()
        print(f"  ✓ Export successful")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return
    
    # Phase 2: Import
    try:
        mod = import_func(prog)
        print(f"  ✓ Import successful")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return
    
    # Phase 3: Compilation
    try:
        ex = relax.build(mod, target="llvm")
        print(f"  ✓ Compilation successful")
    except Exception as e:
        print(f"  ✗ Compilation failed: {e}")
```

### 2. Optimization Level Scanner

```python
def find_working_opt_level(mod):
    """Find the highest working optimization level."""
    from tvm import relax, transform
    
    for opt_level in [0, 1, 2, 3]:
        try:
            with transform.PassContext(opt_level=opt_level):
                ex = relax.build(mod, target="llvm")
            print(f"✓ Works at opt_level={opt_level}")
            return opt_level
        except Exception as e:
            print(f"✗ Fails at opt_level={opt_level}: {str(e)[:100]}")
    
    return None
```

---

## Common Error Patterns

### 1. Type Mismatch Errors

**Error:**
```
Check failed: x->dtype == y->dtype (float32 vs. bool)
```

**Cause:** Operation expects same types but receives mixed types

**Fix:** Cast boolean to float32 before the operation

---

### 2. Boolean Type Inference Errors

**Error:**
```
Cannot decide min_value for typebool
```

**Cause:** TVM's type inference encounters boolean in arithmetic context

**Fix:** Replace boolean operations or cast to numeric type

---

### 3. Dynamic Shape Errors

**Error:**
```
Cannot infer shape for dynamic operation
```

**Cause:** TVM requires static shapes for certain operations

**Fix:** Use `torch.export` with static shapes or add shape constraints

---

## Best Practices

1. **Test components individually** before integration
2. **Save IR dumps** for offline analysis
3. **Use `opt_level=0`** first to rule out optimization issues
4. **Check PyTorch export** - ensure it produces valid IR
5. **Centralize fixes** in `tvm_custom_ops.py` when possible
6. **Document findings** for future reference

---

## References

- **TVM PassContext API**: https://tvm.apache.org/docs/reference/api/python/transform.html
- **TVM Relax Build**: https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.build
- **TVM Type System**: https://tvm.apache.org/docs/reference/api/python/relax/op.html#tvm.relax.op.astype
- **Custom Passes**: https://tvm.apache.org/docs/arch/pass_infra.html
- **PyTorch Export**: https://pytorch.org/docs/stable/export.html

---

## Related Documentation

- [`BOOLEAN_PARAMETER_INVESTIGATION.md`](./BOOLEAN_PARAMETER_INVESTIGATION.md) - Specific case study of boolean parameter issues
- [`TVM_DEBUGGING_LESSONS.md`](./TVM_DEBUGGING_LESSONS.md) - General TVM debugging lessons
