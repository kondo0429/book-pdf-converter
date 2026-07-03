"""
Model directory for Real-ESRGAN models.

Models are not included in git but are bundled when you pip install.
See .gitkeep for setup instructions.
"""

import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files


def get_models_dir() -> Path:
    """Get the path to the models directory."""
    return Path(files(__package__))


def find_model(prefer_coreml: bool = True) -> Path | None:
    """
    Find a model file in the models directory.

    Args:
        prefer_coreml: If True, prefer .mlpackage over .pth

    Returns:
        Path to model file, or None if not found
    """
    models_dir = get_models_dir()

    if prefer_coreml:
        # Look for CoreML first (Mac)
        for pattern in ["*.mlpackage", "*.pth", "*.onnx"]:
            matches = list(models_dir.glob(pattern))
            if matches:
                return matches[0]
    else:
        # Look for PyTorch first
        for pattern in ["*.pth", "*.mlpackage", "*.onnx"]:
            matches = list(models_dir.glob(pattern))
            if matches:
                return matches[0]

    return None
