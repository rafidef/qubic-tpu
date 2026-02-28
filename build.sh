#!/bin/bash
# Build the TPU miner into a standalone Linux binary using PyInstaller
# Run this on the TPU machine (Linux)

set -e

echo "=== Building Qubic TPU Miner Binary ==="

# Install dependencies
pip3 install numpy pyinstaller
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_release.html 2>/dev/null || echo "JAX TPU install skipped (install manually if needed)"

# Build single-file binary
pyinstaller \
    --onefile \
    --name qli-runner \
    --hidden-import=jax \
    --hidden-import=jax.numpy \
    --hidden-import=numpy \
    --hidden-import=qubic_k12 \
    --hidden-import=qubic_keys \
    --hidden-import=qubic_score \
    tpu_miner.py

echo ""
echo "=== Build complete ==="
echo "Binary: dist/qli-runner"
echo ""
echo "Copy dist/qli-runner next to qli-Client and use this appsettings.json:"
echo '{'
echo '  "ClientSettings": {'
echo '    "accessToken": "YOUR_TOKEN",'
echo '    "trainer": {'
echo '      "cpu": true,'
echo '      "gpu": false,'
echo '      "cpuThreads": 4,'
echo '      "customRunner": true'
echo '    }'
echo '  }'
echo '}'
