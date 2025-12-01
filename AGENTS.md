# Agent Strategy and Guidelines

## Environment
- **Python Version**: Python 3.13
- **Launcher**: Always use `py -3.13` to invoke Python and Pip.
  - Run scripts: `py -3.13 script.py`
  - Install packages: `py -3.13 -m pip install package_name`

## Project Development Strategy: SAM3 to TVM Port

### 1. Exploration & Analysis
- **Goal**: Understand the SAM3 model architecture and identify potential roadblocks for TVM compilation.
- **Actions**:
  - Load the SAM3 model using the official codebase.
  - Inspect the model structure for custom operations (e.g., Triton kernels, RoPE, GridSample).
  - Identify dependencies that might be hard to port (e.g., `flash_attn`, `triton` on non-NVIDIA hardware).

### 2. Environment Setup
- Ensure all necessary dependencies are installed for Python 3.13.
- Mock or replace dependencies that are not strictly needed for model tracing/export but cause import errors (e.g., `triton` on macOS).

### 3. Model Export & Tracing
- **Goal**: Obtain a graph representation of the model.
- **Tools**: `torch.export` (Dynamo) is preferred for modern PyTorch to TVM workflows, but `torch.jit.trace` might be used as a fallback.
- **Strategy**:
  - Trace individual components (Image Encoder, Prompt Encoder, Mask Decoder) separately if the full model is too complex.
  - Handle dynamic shapes if necessary.

### 4. TVM Translation & Compilation
- **Goal**: Import the PyTorch graph into TVM Relax.
- **Actions**:
  - Use `tvm.relax.frontend.torch.from_exported_program`.
  - Implement missing op converters in TVM if encountered.
  - For custom ops like RoPE or GridSample, ensure TVM has corresponding implementations or write TIR functions.
  - Address `triton` kernels by replacing them with TVM equivalents or generic PyTorch ops that TVM can handle.

### 5. Verification & Optimization
- **Goal**: Ensure the TVM compiled model produces the same output as the PyTorch model.
- **Actions**:
  - Run inference on both models with the same inputs.
  - Compare outputs using `numpy.testing.assert_allclose`.
  - Tune performance using TVM's tuning capabilities (MetaSchedule).

## Notes
- **SAM3 Specifics**:
  - Uses `flash_attn` and `triton` which are NVIDIA-specific. These need to be bypassed or replaced for a generic TVM port, especially for running on macOS (Metal) or CPU.
  - The `explore_model.py` script is used to verify the model structure and imports.
