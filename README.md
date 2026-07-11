# Book PDF Converter

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[日本語](README.ja.md)

**BookDrive-scan-compatible fork of [Book PDF Converter](https://github.com/robios/book-pdf-converter)**

A tool that converts scanned book PDFs into high-quality documents comparable to digital books, using AI image processing and advanced image processing techniques. For more details, see the original [Book PDF Converter](https://github.com/robios/book-pdf-converter).

## Installation

### Requirements

- Python 3.10–3.13 (3.14+ not supported by dependencies)
- C compiler (for building Cython extensions)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (used for page number detection)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/kondo0429/book-pdf-converter.git
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

### JPEG Folder Processing

Apply the same processing pipeline to a folder of scanned JPEG pages instead of a PDF. Input files are expected to be numbered like `000.JPG, 001.JPG, ...` in ascending order; gaps in the numbering are allowed (the file number keeps the odd/even page grouping aligned). Each output JPEG keeps its input's file name and pixel dimensions (the processed page is fitted onto a white canvas of the input's size).

```bash
# Process all JPEGs in a folder
book-jpeg-converter input_dir/ output_dir/

# Skip AI enhancement
book-jpeg-converter input_dir/ output_dir/ --skip-enhancement

# Process specific file(s) instead of a whole folder (--files / -f).
# The folder argument is omitted; only the output folder is given.
book-jpeg-converter output_dir/ --files scans/003.JPG scans/071.JPG
```

With `--files` you can process a specific subset of pages (one or more paths, which may span folders). Because the page number comes from the file number, a subset still keeps the correct left/right grouping (e.g. `071.JPG` stays an even/left page even when processed alone).

`book-jpeg-converter` supports the same processing options as `book-pdf-converter` (deskew, show-through removal, margin whitening, exclusion lists — page 1 corresponds to `000.JPG`), except the PDF-specific ones (`--dpi`, `--pdf-format`); `--jpeg-quality` controls the output file quality (default: 70).

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
  --debug [FILE]        Print per-page debug output (what was detected and
                        which processing was applied on each page); give FILE
                        to write it to a file instead of the console
  --workers N           Number of parallel workers
```

## Differences from Original

This port faithfully reproduces the original C# implementation, with the following intentional differences:

| Change | Description |
|--------|-------------|
| `--bypass-first/last` | Added option to skip deskew/color/crop for cover pages while still applying AI enhancement |
| Deskew controls | Added `--max-deskew-degree` (default 10°, vs. the original's 1° acceptance limit), `--no-deskew`, and `--deskew-exclude-pages`. Note: the Radon-based detector can only measure up to ~7°, so angles beyond that cannot be corrected regardless of this setting |
| Show-through & background removal | On by default: instead of the original's global-linear color adjustment, each page's paper background is estimated locally (morphological closing + blur), the image is flat-field normalized so paper becomes uniformly white, and a contrast stretch (`--bleed-black-point` / `--bleed-white-point`) decides what is content and what is background. The harsh stretch is not used as the final rendering (it would clip glyph fringes and thin the strokes): pixels it keeps dark form an importance mask, the mask is slightly dilated, and inside it gently stretched flat-field tones are composited over a pure-white background - white background and smooth anti-aliased text at the same time. Removes reverse-side ghost text AND non-uniform background color; output is grayscale. Exclude color/photo pages with `--bleed-removal-exclude-pages` (they keep the original color adjustment) or disable with `--no-bleed-removal` |
| Output page size | Output pages keep the source's pixel dimensions: the processed page is fitted (aspect preserved, centered) onto a white canvas matching each extracted source page (PDF, embedded at the extraction DPI so the physical page size is also preserved) or input file (JPEG), instead of the original's fixed 3508-px-high pages |
| Margin whitening | On by default: the four outer margin bands are painted white — everything left of the leftmost text, right of the rightmost text, above the topmost text, and below the bottommost text. A band is cleared only if it touches a page edge and runs to the opposite edge without any text, so text can never be erased; spine shadows and page-edge streaks are removed. Anything sharing rows/columns with text is left untouched. Scan junk that must not block the bands is detected on the pre-adjustment image and excluded from the text detection: dark regions touching the image border (photography stand / scan bed), thin continuous horizontal dark bars (top/bottom edge lines), and thin continuous vertical bars (left/right page-edge and spine/gutter shadows, caught at a fainter threshold; an ink-fraction guard skips real text columns). In addition, two side-margin passes run after the bands: (a) smooth low-contrast binding shadows survive the flat-field as wider lens-shaped smudges (plus faint streaks) that carry dark cores and re-anchor the bands - each dark blob in the outer side margins is painted out when only a small fraction of its pixels are sharp edges (real text is mostly sharp strokes, a smooth smudge is not); (b) fore-edge page-stack shadows (the exposed stack of page edges) can be dark, textured, very faint, or carry a solid black core, and they re-anchor the text extent so the margin bands stop right at them - the contiguous depressed band in the smoothed column brightness is therefore searched starting AT the freshly painted band boundary (the text extent edge) and painted out over the full height (on flat-fielded pages the background is exactly white, so even a very faint band is detected). Before painting, the strip must prove it is junk: it must contain a column with a continuous non-white vertical run that text can never produce (characters always leave gaps) - a strip that is actually the text block is rejected. Inside a painted strip, glyph-sized ink components (page numbers, running heads) are preserved while tall ink cores are removed. Neither pass touches sharp real text, so edge text (a title reaching near the margin), body text, page numbers and running heads are never cut. A text component swallowed by the (dilated) junk mask still keeps its glyph-sized real-ink parts - a page number sitting on a faint show-through column, or a running head merged with a page-edge shadow line, survives, while ink-free halos, tall shadow cores, neighboring-page print (on border-connected junk) and stack-band speckle are still dropped. All side-margin passes share this glyph-ink notion, so they never erase page numbers or running heads. All these structures are physically impossible for print, and photos are unaffected. Finally, a faint-margin cleanup scrubs the outer side zones of very faint low-contrast marks (fold/edge shadow lines, show-through speckle, tick/dash marks) that slip under every dark threshold: ONLY faint pixels (gray ~160-250) are eligible, so dark text is never touched and no character can be erased even where a body-text column runs to the page edge; the faint anti-aliasing halos of real text are additionally spared wherever dark ink is locally dense, and photos are spared, so only the isolated faint marks are removed. The median cleared band width on each side is then re-granted as page margins around the final crop (replacing `--margin-percent` in this mode), and the same top/bottom margins are shared by odd and even pages so their text starts at the same height. `--no-margin-whitening` / `--margin-pad` |
| Group crop region | Odd and even pages each get a shared crop rectangle. The original uses the **median** of the per-page text bounding boxes (after Tukey-fence outlier rejection, k=1.5); this places the right edge at the median text extent, so any page whose text reaches the common (gutter-side) maximum has its outermost vertical column sliced off — the "right-edge characters cut" case. This port keeps the same outlier rejection (figure/junk-page bboxes are still dropped) but takes the **envelope** (min left, min top, max right, max bottom) of the remaining inliers instead. A shared crop must contain every page's text to avoid clipping any of it, so the correct boundary is the inlier extreme, not the median; no inlier page loses text. The envelope always contains the median rectangle, so nothing that survived before can newly disappear. The per-page text bounding box feeding this also differs: the C# detector blanks a 1% border before measuring (edge-noise guard), which caps the measurable extent ~50px inside the image and hides text that legitimately runs close to the page edge (gutter-side body text) — its outermost column would then be sliced by the crop. Since the box is measured after margin whitening has already cleaned the margins, only a 2px border is blanked (the full 1% is kept with `--no-margin-whitening`) |
| Deskew border exclusion | Angle detection ignores the outer 6% of the left/right edges and 10% of the top/bottom edges. The horizontal exclusion keeps dark spine shadows / page edges (long straight bars) from dominating the Radon projection on sparse pages; the vertical exclusion keeps running titles / footers (long horizontal text lines that page curvature can leave level while the body is skewed) from masking the body skew. Long straight line segments (chart axes and trend lines, table rules, box borders) are also erased from the detection input via a Hough transform - such a line produces a sharp Radon peak at its own angle and could otherwise rotate the whole page (text never forms long straight segments, and the page itself is untouched) |
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

This project is a fork of [Book PDF Converter](https://github.com/robios/book-pdf-converter), which is itself a Python/Cython port of [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) by [Daiyuu Nobori](https://github.com/dnobori).

The Radon transform algorithm for deskewing was ported from [ImageMagick](https://imagemagick.org/)'s `MagickCore/shear.c` (Apache 2.0 License).

## Related Projects

- [Book PDF Converter](https://github.com/robios/book-pdf-converter) - Upstream project this repo is forked from
- [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) - Original C# implementation
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI image enhancement model
- [ImageMagick](https://github.com/ImageMagick/ImageMagick) - Source of Radon transform algorithm
