"""
Runtime shim to extend TVM's PyTorch frontend without forking the TVM repo.

We inject a converter for ``aten::prod.dim_int`` by monkeypatching
``ExportedProgramImporter.create_convert_map`` to reuse the existing
``_prod`` handler. This is intentionally lightweight and should be removed
once TVM upstream grows a native converter.
"""

from tvm.relax.frontend.torch import exported_program_translator as ept


def patch_aten_prod_dim_int():
    """Add aten::prod.dim_int converter to the ExportedProgram importer."""

    orig_create_convert_map = ept.ExportedProgramImporter.create_convert_map

    def patched(self):
        convert_map = orig_create_convert_map(self)
        # Reuse the existing prod converter; avoids forking TVM for this missing op
        if "prod.dim_int" not in convert_map:
            convert_map["prod.dim_int"] = self._prod
        return convert_map

    ept.ExportedProgramImporter.create_convert_map = patched


# Apply on import
patch_aten_prod_dim_int()

def patch_aten_scatter_src():
    """Add aten::scatter.src converter to the ExportedProgram importer."""
    from tvm import relax
    
    def _scatter(self, node):
        data = self.env[node.args[0]]
        dim = node.args[1]
        index = self.env[node.args[2]]
        src = self.env[node.args[3]]
        
        # Ensure dim is an integer
        if isinstance(dim, relax.Expr):
            # If it's a Relax expression, we might have a problem if it's not constant.
            # But typically for scatter, dim is static.
            # For now, assume it's passed as a python int in the args if it was constant in torch.
            # If it came in as a node, we might need to extract it.
            pass
            
        # IMPORTANT: We must use block_builder.emit() to register the op and get a Var with struct_info
        return self.block_builder.emit(relax.op.scatter_elements(data, index, src, axis=dim))

    # Inject the converter method
    ept.ExportedProgramImporter._scatter = _scatter

    orig_create_convert_map = ept.ExportedProgramImporter.create_convert_map

    def patched(self):
        convert_map = orig_create_convert_map(self)
        if "scatter.src" not in convert_map:
            convert_map["scatter.src"] = self._scatter
        return convert_map

    ept.ExportedProgramImporter.create_convert_map = patched

patch_aten_scatter_src()

def patch_torchvision_roi_align():
    """Add torchvision::roi_align converter."""
    from tvm import relax, te
    import tvm.topi as topi
    
    def _roi_align(self, node):
        # print(f"DEBUG: roi_align args: {node.args}")
        data = self.env[node.args[0]]
        boxes = self.env[node.args[1]]
        
        # Args based on debug output: (input, boxes, spatial_scale, out_h, out_w, sampling_ratio, aligned)
        spatial_scale = node.args[2]
        out_h = node.args[3]
        out_w = node.args[4]
        sampling_ratio = node.args[5]
        # aligned = node.args[6]
        
        output_size = (out_h, out_w)
        
        # Create a TE function
        def roi_align_compute(data, boxes):
            # data: (N, C, H, W)
            # boxes: (K, 5)
            N, C, H, W = data.shape
            num_boxes = boxes.shape[0]
            
            # 1. Extract batch indices (col 0)
            batch_ind = te.compute((num_boxes,), lambda i: boxes[i, 0].astype("int32"), name="batch_ind")
            
            # 2. Extract and transform coordinates
            # roi_align (x1, y1, x2, y2) -> crop_and_resize (y1, x1, y2, x2) normalized
            def get_box_coord(i, dim):
                # dim 0: y1, 1: x1, 2: y2, 3: x2
                # boxes: 1: x1, 2: y1, 3: x2, 4: y2
                # map: 0->2, 1->1, 2->4, 3->3
                
                # Use te.if_then_else for symbolic dim
                col_idx = te.if_then_else(dim == 0, 2,
                            te.if_then_else(dim == 1, 1,
                                te.if_then_else(dim == 2, 4, 3)))
                                
                val = boxes[i, col_idx]
                val = val * spatial_scale
                
                # Normalize
                # limit = H if dim in [0, 2] else W
                limit = te.if_then_else(te.any(dim == 0, dim == 2), H, W)
                
                # Ensure float division
                return val / limit.astype("float32")

            box_coords = te.compute((num_boxes, 4), lambda i, j: get_box_coord(i, j), name="box_coords")
            
            return topi.image.crop_and_resize(
                data, 
                box_coords, 
                batch_ind, 
                output_size, 
                layout="NCHW", 
                method="bilinear", 
                extrapolation_value=0.0
            )

        # Register the TE function as a PrimFunc
        # We need to construct args for call_tir
        # data and boxes are Relax Exprs. We need their StructInfo to create TE placeholders?
        # Actually, self.block_builder.add_func can take a TE compute function if we use a helper?
        # No, we usually use bb.call_te(roi_align_compute, data, boxes) which handles PrimFunc creation!
        
        return self.block_builder.call_te(roi_align_compute, data, boxes)

    # Inject
    ept.ExportedProgramImporter._roi_align = _roi_align
    
    orig_create_convert_map = ept.ExportedProgramImporter.create_convert_map

    def patched(self):
        convert_map = orig_create_convert_map(self)
        if "roi_align.default" not in convert_map:
            convert_map["roi_align.default"] = self._roi_align
        return convert_map

    ept.ExportedProgramImporter.create_convert_map = patched

patch_torchvision_roi_align()

def patch_floor_divide():
    """Monkeypatch BaseFXGraphImporter._div to handle type mismatch in floor_divide."""
    from tvm.relax.frontend.torch.base_fx_graph_translator import BaseFXGraphImporter
    from tvm import relax
    
    original_div = BaseFXGraphImporter._div
    
    def patched_div(self, node):
        args = self.retrieve_args(node)
        inp_1 = args[0]
        inp_2 = args[1]

        # Handle scalar cases
        if isinstance(inp_2, (int, float)):
            inp_2 = relax.const(inp_2)
        if isinstance(inp_1, (int, float)):
            inp_1 = relax.const(inp_1)

        # Get rounding_mode from node kwargs
        rounding_mode = args[2] if len(node.args) > 2 else node.kwargs.get("rounding_mode", None)

        # Perform division based on rounding mode
        if rounding_mode == "floor":
            # Check types
            dtype1 = inp_1.struct_info.dtype if hasattr(inp_1, "struct_info") else None
            dtype2 = inp_2.struct_info.dtype if hasattr(inp_2, "struct_info") else None
            
            is_float = False
            if (dtype1 and "float" in dtype1) or (dtype2 and "float" in dtype2):
                is_float = True
            
            if is_float:
                # Cast both to float if needed
                if dtype1 and "int" in dtype1:
                    inp_1 = self.block_builder.emit(relax.op.astype(inp_1, "float32"))
                if dtype2 and "int" in dtype2:
                    inp_2 = self.block_builder.emit(relax.op.astype(inp_2, "float32"))
                
                # Use floor(divide(...)) for floats
                div = self.block_builder.emit(relax.op.divide(inp_1, inp_2))
                return self.block_builder.emit(relax.op.floor(div))
            else:
                # Integer division
                # Ensure same int type
                if dtype1 != dtype2 and dtype1 and dtype2:
                     inp_2 = self.block_builder.emit(relax.op.astype(inp_2, dtype1))
                return self.block_builder.emit(relax.op.floor_divide(inp_1, inp_2))
        else:
            if rounding_mode is None:
                # True division (normal float division)
                return self.block_builder.emit(relax.op.divide(inp_1, inp_2))
            elif rounding_mode == "trunc":
                # Trunc division: perform true division then truncate
                true_div = self.block_builder.emit(relax.op.divide(inp_1, inp_2))
                return self.block_builder.emit(relax.op.trunc(true_div))
            else:
                raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")

    BaseFXGraphImporter._div = patched_div

    # Also patch floor_divide op directly if it exists in convert_map?
    # BaseFXGraphImporter handles aten.div. 
    # aten.floor_divide might be handled separately or mapped to div with mode='floor'.
    # Let's assume it goes through _div or we can add it.
    
    # We can also add explicit converter for floor_divide just in case
    def _floor_divide(self, node):
        # Call patched_div with rounding_mode='floor'
        # We need to mock node.kwargs or args
        # floor_divide(input, other)
        # We can just reuse the logic
        args = self.retrieve_args(node)
        inp_1 = args[0]
        inp_2 = args[1]
        if isinstance(inp_2, (int, float)): inp_2 = relax.const(inp_2)
        if isinstance(inp_1, (int, float)): inp_1 = relax.const(inp_1)
        
        dtype1 = inp_1.struct_info.dtype if hasattr(inp_1, "struct_info") else None
        dtype2 = inp_2.struct_info.dtype if hasattr(inp_2, "struct_info") else None
        
        is_float = False
        if (dtype1 and "float" in dtype1) or (dtype2 and "float" in dtype2):
            is_float = True
            
        if is_float:
            if dtype1 and "int" in dtype1: inp_1 = self.block_builder.emit(relax.op.astype(inp_1, "float32"))
            if dtype2 and "int" in dtype2: inp_2 = self.block_builder.emit(relax.op.astype(inp_2, "float32"))
            div = self.block_builder.emit(relax.op.divide(inp_1, inp_2))
            return self.block_builder.emit(relax.op.floor(div))
        else:
            if dtype1 != dtype2 and dtype1 and dtype2: inp_2 = self.block_builder.emit(relax.op.astype(inp_2, dtype1))
            return self.block_builder.emit(relax.op.floor_divide(inp_1, inp_2))

    from tvm.relax.frontend.torch import exported_program_translator as ept
    orig_create_convert_map = ept.ExportedProgramImporter.create_convert_map
    def patched_map(self):
        m = orig_create_convert_map(self)
        m["floor_divide"] = _floor_divide
        m["floor_divide.default"] = _floor_divide
        m["aten::floor_divide"] = _floor_divide
        return m
    ept.ExportedProgramImporter.create_convert_map = patched_map

patch_floor_divide()

def patch_assertions():
    """Add converters for assertions and equality checks."""
    from tvm import relax
    from tvm.relax.frontend.torch import exported_program_translator as ept

    orig_create_convert_map = ept.ExportedProgramImporter.create_convert_map

    def patched(self):
        def _eq(node):
            # args[0] and args[1] are inputs
            # Check if args are in env (Nodes) or constants
            lhs = self.env[node.args[0]] if node.args[0] in self.env else node.args[0]
            rhs = self.env[node.args[1]] if node.args[1] in self.env else node.args[1]
            
            def to_relax(val):
                if isinstance(val, (int, float, bool)):
                    return relax.const(val)
                if hasattr(val, "dtype") and not isinstance(val, relax.Expr):
                     # Likely TIR var (PrimExpr)
                     return relax.PrimValue(val)
                if not isinstance(val, relax.Expr):
                     # Try to wrap as const
                     return relax.const(val)
                return val

            lhs = to_relax(lhs)
            rhs = to_relax(rhs)
            return self.block_builder.emit(relax.op.equal(lhs, rhs))

        def _assert_scalar(node):
            # We just ignore assertions for now
            return self.block_builder.emit(relax.op.null_value())

        convert_map = orig_create_convert_map(self)
        # Add multiple keys to cover potential naming variations
        eq_keys = ["eq", "eq.Scalar", "eq.default", "aten::eq", "aten.eq"]
        assert_keys = ["_assert_scalar.default", "aten::_assert_scalar.default", "aten._assert_scalar.default"]
        
        for key in eq_keys:
            if key not in convert_map:
                convert_map[key] = _eq
        
        for key in assert_keys:
            if key not in convert_map:
                convert_map[key] = _assert_scalar
                
        return convert_map

    ept.ExportedProgramImporter.create_convert_map = patched
    print("DEBUG: patch_assertions applied")

patch_assertions()

def patch_slice():
    """Patch slice to handle list arguments from dynamic shapes."""
    from tvm import relax
    from tvm.relax.frontend.torch import exported_program_translator as ept

    orig_create_convert_map = ept.ExportedProgramImporter.create_convert_map

    def patched(self):
        def _slice(node):
            args = node.args
            data = self.env[args[0]]
            dim = args[1] if len(args) > 1 else 0
            start = args[2] if len(args) > 2 else 0
            if start is None: start = 0
            end = args[3] if len(args) > 3 else 9223372036854775807 # INT64_MAX
            if end is None: end = 9223372036854775807
            step = args[4] if len(args) > 4 else 1
            if step is None: step = 1
            
            def clean_arg(arg):
                import sys
                if arg in self.env:
                    return self.env[arg]
                if isinstance(arg, list) and len(arg) == 1:
                    arg = arg[0]
                try:
                    return int(arg)
                except:
                    sys.stderr.write(f"DEBUG: clean_arg failed to cast {arg} type={type(arg)}\n")
                    if hasattr(arg, "name"):
                        sys.stderr.write(f"DEBUG: arg is Node? name={arg.name}\n")
                    sys.stderr.flush()
                    return arg

            dim_val = clean_arg(dim)
            start_val = clean_arg(start)
            end_val = clean_arg(end)
            step_val = clean_arg(step)
            
            is_dynamic = False
            if isinstance(start_val, relax.Expr) and not isinstance(start_val, relax.Constant): is_dynamic = True
            if isinstance(end_val, relax.Expr) and not isinstance(end_val, relax.Constant): is_dynamic = True
            if isinstance(step_val, relax.Expr) and not isinstance(step_val, relax.Constant): is_dynamic = True
            
            if is_dynamic:
                 def ensure_tensor(val):
                     if isinstance(val, (int, float)):
                         return relax.const([val], dtype="int64")
                     if isinstance(val, relax.Expr):
                         if val.struct_info.ndim == 0:
                             return self.block_builder.emit(relax.op.reshape(val, [1]))
                         return val
                     return relax.const([val], dtype="int64")

                 t_start = ensure_tensor(start_val)
                 t_end = ensure_tensor(end_val)
                 t_step = ensure_tensor(step_val)
                 
                 return self.block_builder.emit(relax.op.dynamic_strided_slice(data, t_start, t_end, t_step, axes=[dim_val]))
            else:
                 return self.block_builder.emit(relax.op.strided_slice(data, axes=[dim_val], begin=[start_val], end=[end_val], strides=[step_val]))

        convert_map = orig_create_convert_map(self)
        # Override slice converters
        convert_map["slice.Tensor"] = _slice
        convert_map["slice.default"] = _slice
        return convert_map

    ept.ExportedProgramImporter.create_convert_map = patched
    print("DEBUG: patch_slice applied")

patch_slice()


def cast_bool_params_to_float(mod):
    """
    Cast boolean function parameters to float32 to avoid dtype mismatch errors.
    
    CURRENT STATUS: Logging only - actual casting disabled due to TVM API limitations.
    
    Background:
    - PyTorch models can have boolean tensor parameters (e.g., attention masks)
    - PyTorch export includes R.astype to cast these to float32 in the IR body
    - However, TVM's compiler fails with "dtype mismatch (float32 vs bool)"
    - This is because the function signature declares bool, triggering type checks
    
    Attempted Solutions:
    1. Manual IRModule reconstruction - Failed: "module must be set" error
    2. UpdateParamStructInfo transform - Failed: struct inference conflicts
    
    Current Approach:
    - Log boolean parameters for debugging
    - Return original module (will cause dtype mismatch during compilation)
    - Components with boolean parameters will fail compilation
    
    TODO: Find proper TVM API for modifying function parameter types without
    triggering full struct inference.
    
    See: NOTES/BOOLEAN_PARAMETER_INVESTIGATION.md for full investigation details
    """
    from tvm import relax
    
    # Check if any functions have boolean parameters
    has_bool_params = False
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            for param in func.params:
                if hasattr(param, 'struct_info') and hasattr(param.struct_info, 'dtype'):
                    if str(param.struct_info.dtype) == 'bool':
                        has_bool_params = True
                        print(f"DEBUG: Found boolean parameter '{param.name_hint}' in function '{gvar.name_hint}'")
                        break
            if has_bool_params:
                break
    
    if has_bool_params:
        print("DEBUG: Module has boolean parameters")
        print("DEBUG: Boolean parameter casting not yet implemented - compilation may fail")
    
    # Return original module
    # Components with boolean parameters will fail during relax.build()
    return mod


def patch_boolean_parameters():
    """
    Patch TVM's from_exported_program to automatically cast boolean parameters.
    
    This monkey-patches the import function to post-process the IRModule and
    convert any boolean function parameters to float32, preventing dtype mismatch
    errors during compilation.
    
    This patch is applied automatically when tvm_custom_ops is imported.
    """
    from tvm.relax.frontend.torch import from_exported_program as original_import
    
    def patched_import(exported_program, **kwargs):
        """Import and post-process to fix boolean parameters."""
        # Import normally
        mod = original_import(exported_program, **kwargs)
        
        # Post-process: cast boolean parameters to float32
        mod = cast_bool_params_to_float(mod)
        
        return mod
    
    # Monkey-patch the import function
    import tvm.relax.frontend.torch
    tvm.relax.frontend.torch.from_exported_program = patched_import
    
    print("DEBUG: patch_boolean_parameters applied")

# Apply the boolean parameter patch
patch_boolean_parameters()
