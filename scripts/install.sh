#!/usr/bin/env bash
# Linux install: reads the CUDA driver version and picks the matching torch wheel index. No lock file (dependencies resolved per machine).
set -e
cd "$(dirname "$0")/.."
command -v uv >/dev/null || pip install -q uv
[ -d .venv ] || uv venv --python 3.12 .venv 2>/dev/null || uv venv .venv
CUDA=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$" || echo "0")
MAJ=${CUDA%%.*}; MIN=${CUDA#*.}; MIN=${MIN%%.*}
if [ "$MAJ" -ge 13 ]; then IDX=cu130; elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then IDX=cu128; elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then IDX=cu126; elif [ "$MAJ" -eq 12 ]; then IDX=cu121; else IDX=cpu; fi
ARCH=$(uname -m); [ "$ARCH" = "aarch64" ] && [ "$IDX" != "cu130" ] && IDX=cu130   # ARM machines (e.g., DGX Spark) only have cu130 wheels
echo "CUDA driver $CUDA -> torch index $IDX ($ARCH)"
# Order matters: install dependencies first and torch last with a forced index, so a later step cannot replace it with the latest PyPI (cu130) build
uv pip install --python .venv -e .
uv pip install --python .venv --reinstall torch torchvision --index-url https://download.pytorch.org/whl/$IDX
uv pip install --python .venv "numpy<2"
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
