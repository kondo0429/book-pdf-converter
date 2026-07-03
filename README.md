# Book PDF Converter

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[日本語](README.ja.md)

**Python/Cython port of [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter)**

A tool that converts scanned book PDFs into high-quality documents comparable to digital books, using AI image processing and advanced image processing techniques. For more details, see the original [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter).

Currently ported up to the original v1.00.

## Installation

### Requirements

- Python 3.10–3.13 (3.14+ not supported by dependencies)
- C compiler (for building Cython extensions)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (used for page number detection)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

# Install dependencies
pip install -r requirements.txt

# Download and setup AI model (automatically converts to CoreML on Mac)
python scripts/setup_model.py

# Build Cython extensions and install (model is included in package)
pip install .
```

### Platform-Specific Setup

<details>
<summary><b>macOS (Optimized for Apple Silicon)</b></summary>

```bash
# Install system dependencies
brew install tesseract tesseract-lang

# Clone and install
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

pip install -r requirements.txt
python scripts/setup_model.py  # Auto-converts to CoreML
pip install .
```

CoreML provides fast inference using the Neural Engine on M1/M2/M3/M4 chips.

</details>

<details>
<summary><b>Ubuntu/Debian (Linux)</b></summary>

```bash
# Install system dependencies
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng
sudo apt install -y build-essential python3-dev  # For Cython

# Clone and install
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

pip install -r requirements.txt
python scripts/setup_model.py
pip install .
```

For CUDA acceleration, ensure [NVIDIA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) are installed.

</details>

<details>
<summary><b>Windows</b></summary>

1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (add to PATH)
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (for Cython)

```bash
# Clone and install
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

pip install -r requirements.txt
python scripts/setup_model.py
pip install .
```

For CUDA acceleration, install [PyTorch with CUDA](https://pytorch.org/get-started/locally/).

</details>

## Usage

### Basic Usage

```bash
# Convert PDF with AI enhancement
book-pdf-converter input.pdf output.pdf

# Skip AI enhancement (faster, for pre-processed scans)
book-pdf-converter input.pdf output.pdf --skip-enhancement
```

### Batch Processing

Convert multiple PDFs in a directory. Folder structure is preserved.

```bash
# Convert all PDFs in a directory
book-pdf-converter-batch input_dir/ output_dir/

# Skip already converted files
book-pdf-converter-batch input_dir/ output_dir/ --skip-existing

# Continue on error
book-pdf-converter-batch input_dir/ output_dir/ --continue-on-error
```

`book-pdf-converter-batch` supports the same options as `book-pdf-converter`, plus `--skip-existing` and `--continue-on-error`.

### Advanced Options

```bash
# Bypass first/last page (for covers)
book-pdf-converter input.pdf output.pdf --bypass-first --bypass-last

# Specify custom model
book-pdf-converter input.pdf output.pdf --model /path/to/model.pth

# Adjust margin percentage (default: 7%)
book-pdf-converter input.pdf output.pdf --margin-percent 5

# Allow larger deskew correction (up to the given angle in degrees)
book-pdf-converter input.pdf output.pdf --max-deskew-degree 10

# Skip deskew for specific pages (1-indexed, ranges allowed)
book-pdf-converter input.pdf output.pdf --deskew-exclude-pages 1,4,7-9

# Disable deskew for all pages
book-pdf-converter input.pdf output.pdf --no-deskew

# Show-through (bleed-through) & background removal is ON by default (grayscale output).
# Exclude color/photo pages (they keep standard color adjustment) or disable globally
book-pdf-converter input.pdf output.pdf --bleed-removal-exclude-pages 5,12-14
book-pdf-converter input.pdf output.pdf --no-bleed-removal

# Tune show-through removal (lower white point = stronger whitening)
book-pdf-converter input.pdf output.pdf --bleed-white-point 195

# Margin whitening: clears the text-free outer margin bands (ON by default)
book-pdf-converter input.pdf output.pdf --no-margin-whitening
book-pdf-converter input.pdf output.pdf --margin-pad 60

```

### Full Options Reference

```
usage: book-pdf-converter [-h] [--model MODEL] [--scale SCALE] [--tile TILE]
                     [--skip-enhancement] [--dpi DPI]
                     [--margin-percent MARGIN_PERCENT] [--bypass-first]
                     [--bypass-last] [--denoise-strength DENOISE_STRENGTH]
                     [--max-deskew-degree MAX_DESKEW_DEGREE] [--no-deskew]
                     [--deskew-exclude-pages DESKEW_EXCLUDE_PAGES]
                     [--no-bleed-removal]
                     [--bleed-removal-exclude-pages BLEED_REMOVAL_EXCLUDE_PAGES]
                     [--ocr-lang OCR_LANG]
                     [--pdf-format {jpeg,png}] [--jpeg-quality JPEG_QUALITY]
                     [--max-pages MAX_PAGES] [--keep-temp] [--quiet]
                     [--workers WORKERS]
                     input output

Positional arguments:
  input                 Input PDF file
  output                Output PDF file

Options:
  -h, --help            Show help message
  --model, -m MODEL     Path to enhancement model (.mlpackage or .pth)
  --scale, -s SCALE     Upscale factor (default: 2)
  --tile, -t TILE       Tile size for enhancement (default: 512)
  --skip-enhancement    Skip AI enhancement
  --dpi DPI             Input DPI for PDF rendering (default: 300)
  --margin-percent PCT  Output margin percentage (default: 7)
  --bypass-first        Skip processing first page (cover)
  --bypass-last         Skip processing last page (back cover)
  --denoise-strength N  Denoising strength for deskew (default: 20, 0 to disable)
  --max-deskew-degree D Max deskew angle to correct in degrees; larger detections
                        are ignored (default: 10)
  --no-deskew           Disable deskew for all pages
  --deskew-exclude-pages PAGES
                        Page numbers (1-indexed) to skip deskew, e.g. "1,4,7-9"
  --no-bleed-removal    Disable show-through/background removal for all pages
                        (enabled by default; output is grayscale)
  --bleed-removal-exclude-pages PAGES
                        Page numbers (1-indexed) to skip show-through removal,
                        e.g. "1,4,7-9" (excluded pages keep standard color
                        adjustment - use for color/photo pages)
  --bleed-bg-ksize N    Show-through removal: background estimation kernel size
                        (default: 151)
  --bleed-black-point N Show-through removal: values <= this become ink/black
                        (default: 115)
  --bleed-white-point N Show-through removal: values >= this become paper/white;
                        lower removes more (default: 205)
  --no-margin-whitening
                        Disable whitening of the text-free outer margin bands
  --margin-pad N        Margin whitening: pixels kept around the detected text
                        extent (default: 40)
  --ocr-lang LANG       Tesseract language codes (default: eng+jpn)
  --pdf-format FMT      Image format in PDF: jpeg or png (default: jpeg)
  --jpeg-quality N      JPEG quality 0-100 (default: 70)
  --max-pages N         Maximum pages to process (for testing)
  --keep-temp           Keep temporary directory
  --quiet, -q           Suppress progress output
  --workers N           Number of parallel workers
```

## Differences from Original

This port faithfully reproduces the original C# implementation, with the following intentional differences:

| Change | Description |
|--------|-------------|
| `--bypass-first/last` | Added option to skip deskew/color/crop for cover pages while still applying AI enhancement |
| Deskew controls | Added `--max-deskew-degree` (default 10°, vs. the original's 1° acceptance limit), `--no-deskew`, and `--deskew-exclude-pages`. Note: the Radon-based detector can only measure up to ~7°, so angles beyond that cannot be corrected regardless of this setting |
| Show-through & background removal | On by default: instead of the original's global-linear color adjustment, each page's paper background is estimated locally (morphological closing + blur), the image is flat-field normalized so paper becomes uniformly white, and a contrast stretch (`--bleed-black-point` / `--bleed-white-point`) maps ink to black and show-through to white. Removes reverse-side ghost text AND non-uniform background color; output is grayscale. Exclude color/photo pages with `--bleed-removal-exclude-pages` (they keep the original color adjustment) or disable with `--no-bleed-removal` |
| Margin whitening | On by default: the four outer margin bands are painted white — everything left of the leftmost text, right of the rightmost text, above the topmost text, and below the bottommost text. A band is cleared only if it touches a page edge and runs to the opposite edge without any text, so text can never be erased; spine shadows and page-edge streaks are removed. Anything sharing rows/columns with text is left untouched. `--no-margin-whitening` / `--margin-pad` |
| Deskew border exclusion | Angle detection ignores the outer 6% of each edge, so dark spine shadows / page edges (long straight bars) don't dominate the Radon projection on sparse pages and cause wrong-direction rotation |
| PDF extraction resize omitted | C# resizes to A4 (2480×3508) at extraction, but both pipelines normalize to internal high-res (4960×7016), making initial resize redundant |
| Deskew | C# uses ImageMagick external binary on high-res images. This port uses Radon transform ported to Cython, detecting angle on original extracted images with denoising, then applying rotation to high-res images |

There may also be minor behavioral differences due to porting errors.

## Troubleshooting

<details>
<summary><b>Cython build fails</b></summary>

Ensure a C compiler is installed:
- **macOS**: `xcode-select --install`
- **Linux**: `sudo apt install build-essential`
- **Windows**: Install Visual Studio Build Tools

</details>


<details>
<summary><b>CUDA out of memory</b></summary>

Try reducing the tile size:
```bash
book-pdf-converter input.pdf output.pdf --tile 256
```

</details>

## License

This project is licensed under **AGPL-3.0** (GNU Affero General Public License v3.0), the same as the original [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter).

See [LICENSE](LICENSE) for the full license text.

## Acknowledgements

This project is a Python/Cython port of [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) by [Daiyuu Nobori](https://github.com/dnobori).

The Radon transform algorithm for deskewing was ported from [ImageMagick](https://imagemagick.org/)'s `MagickCore/shear.c` (Apache 2.0 License).

## Related Projects

- [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) - Original C# implementation
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI image enhancement model
- [ImageMagick](https://github.com/ImageMagick/ImageMagick) - Source of Radon transform algorithm
