import sys
import os
from unittest.mock import MagicMock

# Add external/sam3 to path
sys.path.append(os.path.abspath("external/sam3"))

def mock_module(name):
    m = MagicMock()
    m.__spec__ = MagicMock()
    m.__path__ = []
    # Add version for triton
    if name == "triton":
        m.__version__ = "2.0.0"  # Fake version to satisfy checks
    sys.modules[name] = m
    return m

# Mock triton
triton = mock_module("triton")
mock_module("triton.language")
mock_module("triton.compiler")
mock_module("triton.runtime")
mock_module("triton.runtime.autotuner")
mock_module("triton.runtime.jit")
mock_module("triton.backends")
mock_module("triton.backends.compiler")

# Mock decord
sys.modules["decord"] = MagicMock()
