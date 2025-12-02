"""
Export SAM3 Transformer Decoder with torch.export and optionally import to TVM Relax.
"""

import torch
import mock_setup  # noqa: F401
import tvm_custom_ops  # noqa: F401
from sam3.model_builder import build_sam3_image_model
from sam3.model.decoder import TransformerDecoder

class TransformerDecoderWrapper(torch.nn.Module):
    """Wrapper to handle optional inputs and tuple outputs for export."""
    def __init__(self, decoder: TransformerDecoder, height: int, width: int):
        super().__init__()
        self.decoder = decoder
        self.height = height
        self.width = width

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor,
        reference_boxes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        memory_text: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ):
        # Hardcode spatial_shapes to avoid data-dependent guard issue
        spatial_shapes = torch.tensor([[self.height, self.width]], device=tgt.device, dtype=torch.long)
        
        # Minimal set of inputs for the decoder
        return self.decoder(
            tgt=tgt,
            memory=memory,
            pos=pos,
            reference_boxes=reference_boxes,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            # Defaults for others
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            apply_dac=False,
        )

def build_decoder() -> TransformerDecoder:
    model = build_sam3_image_model(
        checkpoint_path=None,
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=False,
    )
    decoder = model.transformer.decoder
    decoder = decoder.to("cpu").eval()
    return decoder

def prepare_dummy_inputs(
    batch_size: int = 1,
    d_model: int = 256,
    num_queries: int = 300,
    height: int = 72,
    width: int = 72,
):
    torch.manual_seed(0)
    hw = height * width
    
    # tgt: (nq, bs, d_model)
    tgt = torch.randn(num_queries, batch_size, d_model)
    
    # memory: (hw, bs, d_model)
    memory = torch.randn(hw, batch_size, d_model)
    
    # pos: (hw, bs, d_model)
    pos = torch.randn(hw, batch_size, d_model)
    
    # reference_boxes: (nq, bs, 4)
    reference_boxes = torch.rand(num_queries, batch_size, 4)
    
    # spatial_shapes: (bs, 2) - matching decoder expectation for single scale
    spatial_shapes = torch.tensor([[height, width]], dtype=torch.long).repeat(batch_size, 1)
    
    # level_start_index: (1,)
    level_start_index = torch.tensor([0], dtype=torch.long)
    
    # valid_ratios: (bs, 1, 2)
    valid_ratios = torch.ones(batch_size, 1, 2)

    # memory_text: (num_token, bs, d_model)
    num_text_tokens = 16
    memory_text = torch.randn(num_text_tokens, batch_size, d_model)
    
    # text_attention_mask: (bs, num_token)
    text_attention_mask = torch.zeros(batch_size, num_text_tokens, dtype=torch.bool)

    return tgt, memory, pos, reference_boxes, spatial_shapes, level_start_index, valid_ratios, memory_text, text_attention_mask

def export_decoder():
    print("=== Building Transformer Decoder ===")
    decoder = build_decoder()
    # Use 72x72 to match default model config
    height, width = 72, 72
    wrapper = TransformerDecoderWrapper(decoder, height=height, width=width)
    wrapper.eval()
    
    # Pass height/width to prepare_dummy_inputs
    inputs = prepare_dummy_inputs(height=height, width=width)
    
    # Remove spatial_shapes from inputs (it's the 5th element, index 4)
    # inputs tuple: (tgt, memory, pos, reference_boxes, spatial_shapes, level_start_index, valid_ratios, memory_text, text_attention_mask)
    # New inputs for wrapper: (tgt, memory, pos, reference_boxes, level_start_index, valid_ratios, memory_text, text_attention_mask)
    wrapper_inputs = inputs[:4] + inputs[5:]
    
    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        outputs = wrapper(*wrapper_inputs)
    print("✓ Forward pass succeeded")
    
    # outputs is a tuple: (intermediate, intermediate_ref_boxes, presence_logits, presence_feats)
    print(f"  intermediate: {outputs[0].shape}")
    print(f"  ref_boxes: {outputs[1].shape}")
    
    print("\n=== Exporting with torch.export ===")
    
    try:
        exported_program = torch.export.export(
            wrapper,
            wrapper_inputs,
            strict=True,
        )
    except Exception as err:
        print(f"✗ Export failed: {err}")
        import traceback
        traceback.print_exc()
        return None

    torch.export.save(exported_program, "sam3_transformer_decoder_exported.pt2")
    print("✓ Exported to sam3_transformer_decoder_exported.pt2")
    return exported_program

def import_to_tvm(exported_program):
    if exported_program is None:
        return None

    print("\n=== Importing to TVM Relax ===")
    try:
        import tvm
        from tvm.relax.frontend.torch import from_exported_program
    except Exception as err:
        print(f"✗ TVM not available: {err}")
        return None

    # Monkeypatch BaseFXGraphImporter._div to handle type mismatch in floor_divide
    from tvm.relax.frontend.torch.base_fx_graph_translator import BaseFXGraphImporter
    from tvm import relax
    import torch.fx as fx
    
    original_div = BaseFXGraphImporter._div
    
    def patched_div(self, node: fx.Node) -> relax.Var:
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
            # Delegate to original for other cases (or copy logic if needed, but original handles None and trunc)
            # Since original_div expects self to be the instance, and we are calling it as unbound method?
            # No, original_div is a function (unbound method in Py3).
            # But we need to pass 'self'.
            # However, we already consumed args.
            # It's safer to copy the logic for other cases too to avoid re-retrieving args side effects (though retrieve_args should be idempotent).
            
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

    print("\n=== Importing to TVM Relax ===")
    try:
        mod = from_exported_program(
            exported_program,
            keep_params_as_input=False,
        )
        mod.show()
    except Exception as err:
        print(f"✗ TVM import failed: {err}")
        import traceback
        traceback.print_exc()
        return None

    print("✓ Imported decoder to TVM Relax")
    with open("sam3_transformer_decoder_tvm.txt", "w") as f:
        f.write(str(mod))
    print("  Saved TVM IR to sam3_transformer_decoder_tvm.txt")
    return mod

if __name__ == "__main__":
    exported = export_decoder()
    if exported is not None:
        import_to_tvm(exported)
