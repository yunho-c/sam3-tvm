"""
Export SAM3 Transformer Encoder (TransformerEncoderFusion) with torch.export and
optionally import to TVM Relax.

The goal is to exercise the encoder's cross-attention stack without pulling in
the full SAM3 model. Inputs are plain tensors shaped like the real model:
- Vision features: list of one (batch, d_model, H, W) tensor
- Vision positional encodings: matching (batch, d_model, H, W)
- Text prompt embeddings: (batch, prompt_len, d_model)
- Prompt padding mask: (batch, prompt_len) bool
- Vision padding mask: (batch, H, W) bool
"""

import torch
import mock_setup  # noqa: F401 - ensures external deps are mocked before import
from sam3.model_builder import build_sam3_image_model


class TransformerEncoderWrapper(torch.nn.Module):
    """Flatten dict outputs so torch.export produces a tensor tuple."""

    def __init__(self, encoder: torch.nn.Module, feat_size_hw: int):
        super().__init__()
        self.encoder = encoder
        self.feat_size_hw = feat_size_hw

    def forward(
        self,
        vision_feat_seq: torch.Tensor,
        vision_pos_seq: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        # Provide feat_sizes so the encoder reshapes seq-first tensors back to (bs, c, h, w)
        feat_sizes = [(self.feat_size_hw, self.feat_size_hw)]
        out = self.encoder(
            src=[vision_feat_seq],
            src_key_padding_mask=None,
            src_pos=[vision_pos_seq],
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
            prompt_pos=None,
            feat_sizes=feat_sizes,
            encoder_extra_kwargs=None,
        )

        return (
            out["memory"],
            out["padding_mask"],
            out["pos_embed"],
            out["memory_text"],
            out["level_start_index"],
            out["spatial_shapes"],
            out["valid_ratios"],
        )


def build_encoder() -> torch.nn.Module:
    """Create the SAM3 transformer encoder with pretrained weights if present."""

    model = build_sam3_image_model(
        checkpoint_path=None,
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=False,
    )
    encoder = model.transformer.encoder
    encoder = encoder.to("cpu").eval()
    return encoder


def prepare_dummy_inputs(
    batch_size: int = 2,
    d_model: int = 256,
    height: int = 16,
    width: int = 16,
    prompt_len: int = 8,
):
    """Create deterministic dummy tensors matching encoder expectations."""

    torch.manual_seed(0)
    hw = height * width
    # Sequence-first tensors: [HW, B, C]
    vision_feat = torch.randn(hw, batch_size, d_model)
    vision_pos = torch.randn(hw, batch_size, d_model)
    # Prompt is seq-first so that encoder's transpose yields batch-first
    prompt = torch.randn(prompt_len, batch_size, d_model)
    prompt_mask = torch.zeros(batch_size, prompt_len, dtype=torch.bool)
    return vision_feat, vision_pos, prompt, prompt_mask, height


def export_transformer_encoder():
    """Export TransformerEncoderFusion with torch.export."""

    print("=== Building Transformer Encoder ===")
    encoder = build_encoder()
    # height==width for dummy grid; pass through for feat_sizes reconstruction
    dummy_inputs = prepare_dummy_inputs()
    wrapper = TransformerEncoderWrapper(encoder, feat_size_hw=dummy_inputs[-1])
    wrapper.eval()

    inputs = dummy_inputs[:-1]
    print("Dummy inputs prepared:")
    print(f"  vision_feat_seq (HW, B, C): {inputs[0].shape}")
    print(f"  vision_pos_seq  (HW, B, C): {inputs[1].shape}")
    print(f"  prompt          (S, B, C): {inputs[2].shape}")
    print(f"  prompt_mask: {inputs[3].shape}")

    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        outputs = wrapper(*inputs)
    print("✓ Forward pass succeeded")
    for i, t in enumerate(outputs):
        print(f"  output[{i}]: {t if t is None else t.shape}")

    print("\n=== Exporting with torch.export ===")
    try:
        exported_program = torch.export.export(
            wrapper,
            inputs,
            strict=False,
        )
    except Exception as err:  # pragma: no cover - export failure path
        print(f"✗ Export failed: {err}")
        import traceback

        traceback.print_exc()
        return None

    torch.export.save(exported_program, "sam3_transformer_encoder_exported.pt2")
    print("✓ Exported to sam3_transformer_encoder_exported.pt2")
    return exported_program


def import_to_tvm(exported_program):
    """Attempt to import the exported encoder into TVM Relax."""

    if exported_program is None:
        print("No exported program to import")
        return None

    print("\n=== Importing to TVM Relax ===")
    try:
        import tvm
        from tvm.relax.frontend.torch import from_exported_program
    except Exception as err:  # pragma: no cover - import failure path
        print(f"✗ TVM not available: {err}")
        return None

    try:
        mod = from_exported_program(
            exported_program,
            keep_params_as_input=False,
        )
    except Exception as err:  # pragma: no cover - converter failure path
        print(f"✗ TVM import failed: {err}")
        import traceback

        traceback.print_exc()
        return None

    print("✓ Imported encoder to TVM Relax")
    with open("sam3_transformer_encoder_tvm.txt", "w") as f:
        f.write(str(mod))
    print("  Saved TVM IR to sam3_transformer_encoder_tvm.txt")
    return mod


if __name__ == "__main__":
    exported = export_transformer_encoder()
    if exported is not None:
        import_to_tvm(exported)
