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
            
        return relax.op.scatter_elements(data, index, src, axis=dim)

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

        # Get rounding_mode from node kwargs
        rounding_mode = args[2] if len(node.args) > 2 else node.kwargs.get("rounding_mode", None)

        # Perform division based on rounding mode
        if rounding_mode == "floor":
            # Check types and cast if mismatch
            dtype1 = inp_1.struct_info.dtype if hasattr(inp_1, "struct_info") else None
            dtype2 = inp_2.struct_info.dtype if hasattr(inp_2, "struct_info") else None
            
            if dtype1 and dtype2 and dtype1 != dtype2:
                # If one is float and other is int, cast int to float
                if "float" in dtype1 and "int" in dtype2:
                     inp_2 = self.block_builder.emit(relax.op.astype(inp_2, dtype1))
                elif "int" in dtype1 and "float" in dtype2:
                     inp_1 = self.block_builder.emit(relax.op.astype(inp_1, dtype2))
            
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

patch_floor_divide()
