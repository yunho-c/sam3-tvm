#!/bin/bash
# Build TVM native libraries from source
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TVM_DIR="${PROJECT_ROOT}/external/tvm"
BUILD_DIR="${TVM_DIR}/build"

echo "=== Building TVM ==="
echo "TVM source: ${TVM_DIR}"
echo "Build directory: ${BUILD_DIR}"

# Initialize git submodules (required for tvm-ffi and other dependencies)
echo "Initializing git submodules..."
cd "$TVM_DIR"
git submodule update --init --recursive

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_LLVM=ON \
    -DUSE_CUDA=OFF \
    -DUSE_METAL=OFF \
    -DUSE_VULKAN=OFF \
    -DUSE_OPENCL=OFF

# Build using all available cores
cmake --build . --parallel "$(getconf _NPROCESSORS_ONLN)"

echo "=== TVM build complete ==="
echo "Run 'pixi run install-tvm' to install the Python package"
