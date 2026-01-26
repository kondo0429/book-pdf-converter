"""
Image enhancement using Real-ESRGAN models.

Uses CoreML on macOS (Neural Engine) and PyTorch/Spandrel on other platforms.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
import cv2


class BaseEnhancer(ABC):
    """Base class for image enhancement."""

    # Native scale of the model (set by subclass)
    native_scale: int = 4

    def __init__(self, model_path: str | Path, scale: int = 2, tile_size: int = 512):
        self.model_path = Path(model_path)
        self.out_scale = scale  # User's requested output scale
        self.tile_size = tile_size

    @property
    def scale(self) -> int:
        """Output scale (for compatibility)."""
        return self.out_scale

    @abstractmethod
    def _enhance_native(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single image at native scale (internal, no downscaling)."""
        pass

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single image, downscaling to user's requested scale if needed."""
        # Ensure RGB
        was_grayscale = len(image.shape) == 2
        if was_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Run at native scale
        output = self._enhance_native(image)

        # Downscale if user requested scale < native scale
        if self.out_scale < self.native_scale:
            final_h = h * self.out_scale
            final_w = w * self.out_scale
            output = cv2.resize(output, (final_w, final_h), interpolation=cv2.INTER_AREA)

        if was_grayscale:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

        return output

    def enhance_tiled(self, image: np.ndarray, tile_pad: int = 16) -> np.ndarray:
        """
        Enhance image using tiles for large images / limited memory.

        C# faithful implementation with tile_pad (not overlap blending).
        Each tile is padded by tile_pad on each side before inference,
        then the padding is cropped after, avoiding edge artifacts.

        Args:
            image: Input image (H, W) grayscale or (H, W, 3) RGB
            tile_pad: Padding around each tile in pixels (C# default: 16)

        Returns:
            Enhanced image at user's requested scale
        """
        # Ensure 3-channel for processing
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            was_grayscale = True
        else:
            was_grayscale = False

        h, w = image.shape[:2]
        # Always use native_scale for internal processing
        out_h, out_w = h * self.native_scale, w * self.native_scale

        # Output buffer
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # C# approach: tile_size stride with tile_pad padding on each side
        # Tiles don't overlap in the output - padding is for inference only
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                # Tile boundaries in input
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                tile_h = y_end - y
                tile_w = x_end - x

                # Padded region boundaries (clamped to image bounds)
                pad_y_start = max(0, y - tile_pad)
                pad_x_start = max(0, x - tile_pad)
                pad_y_end = min(h, y_end + tile_pad)
                pad_x_end = min(w, x_end + tile_pad)

                # Extract padded tile
                padded_tile = image[pad_y_start:pad_y_end, pad_x_start:pad_x_end]

                # Calculate how much actual padding we got on each side
                actual_pad_top = y - pad_y_start
                actual_pad_left = x - pad_x_start
                actual_pad_bottom = pad_y_end - y_end
                actual_pad_right = pad_x_end - x_end

                # Enhance padded tile at native scale
                enhanced_padded = self._enhance_native(padded_tile)

                # Crop to remove padding from enhanced output (scale up padding amounts)
                crop_top = actual_pad_top * self.native_scale
                crop_left = actual_pad_left * self.native_scale
                crop_bottom = enhanced_padded.shape[0] - actual_pad_bottom * self.native_scale
                crop_right = enhanced_padded.shape[1] - actual_pad_right * self.native_scale
                enhanced_tile = enhanced_padded[crop_top:crop_bottom, crop_left:crop_right]

                # Output coordinates (using native_scale)
                oy = y * self.native_scale
                ox = x * self.native_scale
                oy_end = oy + enhanced_tile.shape[0]
                ox_end = ox + enhanced_tile.shape[1]

                # Place in output
                output[oy:oy_end, ox:ox_end] = enhanced_tile

        # Downscale if user requested scale < native scale (C# approach)
        if self.out_scale < self.native_scale:
            final_h = h * self.out_scale
            final_w = w * self.out_scale
            output = cv2.resize(output, (final_w, final_h), interpolation=cv2.INTER_AREA)

        if was_grayscale:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

        return output


class CoreMLEnhancer(BaseEnhancer):
    """CoreML-based enhancer for macOS (uses Neural Engine)."""

    # Real-ESRGAN models are natively 4x
    native_scale: int = 4

    def __init__(self, model_path: str | Path, scale: int = 2, tile_size: int = 512):
        super().__init__(model_path, scale, tile_size)
        import coremltools as ct
        self.model = ct.models.MLModel(str(self.model_path))

    def _enhance_native(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single image using CoreML at native 4x scale."""
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Preprocess: HWC to CHW, normalize to [0, 1]
        img = image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Run inference
        result = self.model.predict({"input": img})

        # Get output
        if "output" in result:
            output = result["output"]
        else:
            output = list(result.values())[0]

        # Postprocess: CHW to HWC
        if output.ndim == 4:
            output = output[0]
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        return output


class PyTorchEnhancer(BaseEnhancer):
    """PyTorch/Spandrel-based enhancer for Linux/Windows."""

    def __init__(self, model_path: str | Path, scale: int = 2, tile_size: int = 512,
                 device: Optional[str] = None):
        super().__init__(model_path, scale, tile_size)

        import torch
        import spandrel

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.model = spandrel.ModelLoader().load_from_file(str(self.model_path))
        self.model = self.model.to(self.device).eval()
        # Use model's native scale (usually 4x for Real-ESRGAN)
        self.native_scale = self.model.scale

    def _enhance_native(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single image using PyTorch at native scale."""
        import torch

        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Preprocess
        img = image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(tensor)

        # Postprocess
        output = output.squeeze(0).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        return output


def create_enhancer(
    model_path: str | Path,
    scale: int = 2,
    tile_size: int = 512,
    force_backend: Optional[str] = None,
) -> BaseEnhancer:
    """
    Create an enhancer using the appropriate backend for the platform.

    Args:
        model_path: Path to model file (.mlpackage for CoreML, .pth for PyTorch)
        scale: Upscaling factor
        tile_size: Tile size for processing
        force_backend: Force 'coreml' or 'pytorch' backend

    Returns:
        Enhancer instance
    """
    model_path = Path(model_path)

    if force_backend == 'coreml' or (force_backend is None and sys.platform == 'darwin'):
        if model_path.suffix == '.pth':
            raise ValueError("CoreML backend requires .mlpackage model. Convert with convert_to_coreml.py")
        return CoreMLEnhancer(model_path, scale, tile_size)
    else:
        if str(model_path).endswith('.mlpackage'):
            raise ValueError("PyTorch backend requires .pth model")
        return PyTorchEnhancer(model_path, scale, tile_size)


def enhance_image(
    image: np.ndarray,
    model_path: str | Path,
    scale: int = 2,
    tile_size: int = 512,
) -> np.ndarray:
    """
    Convenience function to enhance a single image.

    Args:
        image: Input image array
        model_path: Path to model file
        scale: Upscaling factor
        tile_size: Tile size (0 for no tiling)

    Returns:
        Enhanced image array
    """
    enhancer = create_enhancer(model_path, scale, tile_size)

    if tile_size > 0:
        return enhancer.enhance_tiled(image)
    else:
        return enhancer.enhance(image)
