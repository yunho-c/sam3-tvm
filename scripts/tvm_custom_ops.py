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
