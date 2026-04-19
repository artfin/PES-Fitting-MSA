#!/bin/bash
# Setup script for PES-Fitting-MSA environment
# Usage: ./setup_env.sh [venv_path]
#
# Example:
#   ./setup_env.sh              # uses ./venv
#   ./setup_env.sh ~/my_venv    # uses custom path

set -e

# Pinned versions for reproducibility and distributed training compatibility
TORCH_VERSION="2.10.0"
CUDA_VERSION="cu128"

VENV_PATH=${1:-venv}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PES-Fitting-MSA Environment Setup ==="
echo "Project: $PROJECT_DIR"
echo "Venv: $VENV_PATH"
echo "PyTorch: ${TORCH_VERSION}+${CUDA_VERSION}"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate
source "$VENV_PATH/bin/activate"
echo "Activated: $(which python)"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (pinned version)
echo "Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}..."
pip install torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install core requirements
echo "Installing core dependencies..."
pip install -r "$PROJECT_DIR/requirements-core.txt"

# Install extxyz from GitHub
echo "Installing extxyz from GitHub..."
pip install git+https://github.com/libAtoms/extxyz.git || echo "Warning: extxyz installation failed (optional)"

# Optional: wandb
read -p "Install wandb for experiment tracking? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install wandb
fi

# Verify installation
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || true
python -c "import torch; print(f'NCCL version: {torch.cuda.nccl.version()}')" 2>/dev/null || true
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "No GPU detected"
echo ""
echo "=== Setup Complete ==="
echo "Activate with: source $VENV_PATH/bin/activate"
echo ""
echo "For distributed training, ensure all nodes show identical versions above."
