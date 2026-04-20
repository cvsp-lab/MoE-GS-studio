# MoDE: uv Environment Setup Guide

## Prerequisites

* CUDA Toolkit 11.8 or higher
* uv ([https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/))

## Environment Creation and Package Installation

```bash
# 1. Create a Python 3.9 virtual environment
uv venv --python 3.9

# 2. Activate
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. Install dependencies
uv pip install -e .

# Or explicitly specify the PyTorch CUDA version:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .
```

## Build CUDA Extensions (after restoring third-party modules)

Refer to [`THIRDPARTY_RECOVERY.md`](THIRDPARTY_RECOVERY.md) for instructions on restoring third-party modules.

```bash
uv pip install submodules/depth-diff-gaussian-rasterization/
uv pip install submodules/simple-knn/
```

## Migration from conda

The existing `requirements.txt` is retained for reference.
