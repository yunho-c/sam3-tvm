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
