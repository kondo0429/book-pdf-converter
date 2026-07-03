#!/usr/bin/env python3
"""
Download and set up Real-ESRGAN model for book-pdf-converter.

This script:
1. Downloads RealESRGAN_x4plus.pth from GitHub releases
2. On Mac: converts to CoreML (.mlpackage) for fast inference
3. Places the model in converter/models/

After running this script, run `pip install .` to bundle the model.

Usage:
    python scripts/setup_model.py
    python scripts/setup_model.py --skip-download  # if you already have the .pth
    python scripts/setup_model.py --pth-only       # skip CoreML conversion
"""

import argparse
import platform
import sys
import urllib.request
from pathlib import Path

# Model URL and paths
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_NAME = "RealESRGAN_x4plus"

# Get project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "converter" / "models"


def download_model(output_path: Path) -> bool:
    """Download the model from GitHub releases."""
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        return True

    print(f"Downloading {MODEL_NAME}.pth...")
    print(f"  From: {MODEL_URL}")
    print(f"  To: {output_path}")

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, output_path, reporthook=report_progress)
        print("\n  Download complete!")
        return True

    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def convert_to_coreml(pth_path: Path, output_path: Path) -> bool:
    """Convert PyTorch model to CoreML."""
    if output_path.exists():
        print(f"CoreML model already exists: {output_path}")
        return True

    print(f"Converting to CoreML...")
    print(f"  Input: {pth_path}")
    print(f"  Output: {output_path}")

    try:
        import torch
        import coremltools as ct
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as e:
        print(f"  Missing dependencies: {e}")
        print("  Install with: pip install torch coremltools basicsr")
        return False

    try:
        # Load PyTorch model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )

        loadnet = torch.load(pth_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        elif 'params' in loadnet:
            keyname = 'params'
        else:
            keyname = None

        if keyname:
            model.load_state_dict(loadnet[keyname], strict=True)
        else:
            model.load_state_dict(loadnet, strict=True)

        model.eval()

        # Trace the model
        input_size = 128
        example_input = torch.randn(1, 3, input_size, input_size)
        print(f"  Tracing model with input shape: {example_input.shape}")
        traced_model = torch.jit.trace(model, example_input)

        # Convert to CoreML with flexible input shape
        print("  Converting to CoreML (this may take a minute)...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input",
                    shape=ct.Shape(
                        shape=(1, 3,
                               ct.RangeDim(lower_bound=32, upper_bound=1024, default=input_size),
                               ct.RangeDim(lower_bound=32, upper_bound=1024, default=input_size))
                    )
                )
            ],
            outputs=[ct.TensorType(name="output")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS13,
        )

        # Save
        mlmodel.save(str(output_path))
        print(f"  CoreML conversion complete!")
        return True

    except Exception as e:
        print(f"  Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Set up Real-ESRGAN model")
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download (use existing .pth file)')
    parser.add_argument('--pth-only', action='store_true',
                        help='Skip CoreML conversion (keep .pth only)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing files')
    args = parser.parse_args()

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pth_path = MODELS_DIR / f"{MODEL_NAME}.pth"
    mlpackage_path = MODELS_DIR / f"{MODEL_NAME}.mlpackage"

    # Remove existing if --force
    if args.force:
        if pth_path.exists():
            pth_path.unlink()
        if mlpackage_path.exists():
            import shutil
            shutil.rmtree(mlpackage_path)

    # Step 1: Download
    if not args.skip_download:
        if not download_model(pth_path):
            sys.exit(1)
    else:
        if not pth_path.exists():
            print(f"Error: {pth_path} not found. Remove --skip-download to download it.")
            sys.exit(1)

    # Step 2: Convert to CoreML (Mac only)
    is_mac = platform.system() == "Darwin"

    if is_mac and not args.pth_only:
        if not convert_to_coreml(pth_path, mlpackage_path):
            print("\nCoreML conversion failed, but .pth file is available.")
            print("You can use PyTorch inference instead.")
    elif not is_mac:
        print("\nNot on Mac - skipping CoreML conversion.")
        print("PyTorch model (.pth) is ready for use.")

    # Summary
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print(f"\nModels directory: {MODELS_DIR}")

    if pth_path.exists():
        size_mb = pth_path.stat().st_size / (1024 * 1024)
        print(f"  - {pth_path.name} ({size_mb:.1f} MB)")

    if mlpackage_path.exists():
        print(f"  - {mlpackage_path.name}/")

    print("\nNext steps:")
    print("  1. Install the package: pip install .")
    print("  2. Run: book-pdf-converter input.pdf output.pdf")
    print("\nThe model will be auto-detected - no need to specify --model!")


if __name__ == '__main__':
    main()
