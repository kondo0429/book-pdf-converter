#!/usr/bin/env python3
"""
JPEG Converter CLI - Convert and enhance a folder of scanned JPEG pages.

Applies the same processing pipeline as book-pdf-converter (crop, AI
enhancement, deskew, show-through removal, margin whitening, crop
unification, OCR alignment) to JPEG files instead of a PDF.

Input files are expected to be numbered like 000.JPG, 001.JPG, ... in
ascending order (gaps allowed - the file number keeps the odd/even page
grouping aligned). Each output JPEG keeps its input's file name.

Usage:
    book-jpeg-converter input_dir/ output_dir/
    book-jpeg-converter input_dir/ output_dir/ --skip-enhancement
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from . import ConversionOptions
from .pipeline import convert_images
from .cli import parse_page_ranges, reconfigure_stdio_utf8
from .models import find_model


def main():
    reconfigure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description="Convert and enhance a folder of scanned JPEG pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  book-jpeg-converter scans/ cleaned/                # auto-detect bundled model
  book-jpeg-converter scans/ cleaned/ --skip-enhancement
  book-jpeg-converter scans/ cleaned/ --bleed-removal-exclude-pages 1,5
        """,
    )

    # Input/Output folders
    parser.add_argument('input_dir', type=str, help='Folder containing input JPEG files (000.JPG, 001.JPG, ...)')
    parser.add_argument('output_dir', type=str, help='Folder for processed JPEG files (same names as input)')

    # Enhancement options
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to enhancement model (.mlpackage for Mac, .pth for Linux/Windows). Auto-detected if installed.')
    parser.add_argument('--scale', '-s', type=int, default=2,
                        help='Upscaling factor (default: 2)')
    parser.add_argument('--tile', '-t', type=int, default=512,
                        help='Tile size for enhancement, 0 for no tiling (default: 512)')
    parser.add_argument('--skip-enhancement', action='store_true',
                        help='Skip AI enhancement')

    # Margin settings
    parser.add_argument('--margin-percent', type=int, default=7,
                        help='Output margin percentage when margin whitening is disabled (default: 7)')

    # Bypass options (skip deskew/color/crop, but keep ESRGAN)
    parser.add_argument('--bypass-first', action='store_true',
                        help='Bypass first page processing (keeps AI enhancement, skips deskew/color/crop)')
    parser.add_argument('--bypass-last', action='store_true',
                        help='Bypass last page processing (keeps AI enhancement, skips deskew/color/crop)')

    # Deskew preprocessing
    parser.add_argument('--denoise-strength', type=int, default=20,
                        help='Non-local means denoising strength for deskew (0=disabled, default: 20)')
    parser.add_argument('--max-deskew-degree', type=float, default=10.0,
                        help='Max deskew angle to correct in degrees; larger detections are ignored (default: 10)')
    parser.add_argument('--no-deskew', action='store_true',
                        help='Disable deskew for all pages')
    parser.add_argument('--deskew-exclude-pages', type=str, default=None,
                        help='Page numbers (1-indexed; 000.JPG is page 1) to skip deskew, e.g. "1,4,7-9"')

    # Show-through (bleed-through) removal - on for all pages by default (grayscale output)
    parser.add_argument('--no-bleed-removal', action='store_true',
                        help='Disable show-through/background removal for all pages')
    parser.add_argument('--bleed-removal-exclude-pages', type=str, default=None,
                        help='Page numbers (1-indexed; 000.JPG is page 1) to skip show-through removal, e.g. "1,4,7-9" '
                             '(they keep standard color adjustment, e.g. color/photo pages)')
    parser.add_argument('--bleed-bg-ksize', type=int, default=151,
                        help='Show-through removal: background estimation kernel size (default: 151)')
    parser.add_argument('--bleed-black-point', type=int, default=115,
                        help='Show-through removal: values <= this become ink/black (default: 115)')
    parser.add_argument('--bleed-white-point', type=int, default=205,
                        help='Show-through removal: values >= this become paper/white; lower removes more (default: 205)')

    # Margin whitening - clear the text-free outer margin bands (on by default).
    parser.add_argument('--no-margin-whitening', action='store_true',
                        help='Disable whitening of the text-free outer margin bands')
    parser.add_argument('--margin-pad', type=int, default=40,
                        help='Margin whitening: pixels kept around the detected text extent (default: 40)')

    # OCR settings
    parser.add_argument('--ocr-lang', type=str, default='eng+jpn',
                        help='Tesseract language codes (default: eng+jpn)')

    # Output options
    parser.add_argument('--jpeg-quality', type=int, default=70,
                        help='Output JPEG quality 0-100 (default: 70)')

    # Debug options
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Maximum pages to process (for testing)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temp directory after processing')

    # Other
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--debug', nargs='?', const='', default=None, metavar='FILE',
                        help='Print per-page debug output (detections and applied processing); '
                             'give FILE to write it to a file instead of the console')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPUs)')

    args = parser.parse_args()

    # Validate input folder
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Input folder not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect model if not specified
    if not args.skip_enhancement and not args.model:
        import platform
        prefer_coreml = platform.system() == "Darwin"
        detected_model = find_model(prefer_coreml=prefer_coreml)
        if detected_model:
            args.model = str(detected_model)
            if not args.quiet:
                print(f"Using bundled model: {detected_model.name}")
        else:
            print("Warning: No model found. Use --model or --skip-enhancement", file=sys.stderr)
            print("To install a model, run: python scripts/setup_model.py", file=sys.stderr)
            print("Continuing without enhancement...", file=sys.stderr)
            args.skip_enhancement = True

    # Parse deskew exclude pages
    deskew_exclude_pages = None
    if args.deskew_exclude_pages:
        try:
            deskew_exclude_pages = parse_page_ranges(args.deskew_exclude_pages)
        except ValueError as e:
            print(f"Error: invalid --deskew-exclude-pages: {e}", file=sys.stderr)
            sys.exit(1)

    # Parse bleed-removal exclude pages
    bleed_removal_exclude_pages = None
    if args.bleed_removal_exclude_pages:
        try:
            bleed_removal_exclude_pages = parse_page_ranges(args.bleed_removal_exclude_pages)
        except ValueError as e:
            print(f"Error: invalid --bleed-removal-exclude-pages: {e}", file=sys.stderr)
            sys.exit(1)

    # Build options
    options_kwargs = dict(
        margin_percent=args.margin_percent,
        bypass_first_page=args.bypass_first,
        bypass_last_page=args.bypass_last,
        max_pages=args.max_pages,
        skip_enhancement=args.skip_enhancement,
        model_path=args.model,
        scale=args.scale,
        tile_size=args.tile,
        keep_temp=args.keep_temp,
        ocr_lang=args.ocr_lang,
        jpeg_quality=args.jpeg_quality,
        denoise_strength=args.denoise_strength,
        max_deskew_degree=args.max_deskew_degree,
        no_deskew=args.no_deskew,
        deskew_exclude_pages=deskew_exclude_pages,
        no_bleed_removal=args.no_bleed_removal,
        bleed_removal_exclude_pages=bleed_removal_exclude_pages,
        bleed_bg_ksize=args.bleed_bg_ksize,
        bleed_black_point=args.bleed_black_point,
        bleed_white_point=args.bleed_white_point,
        disable_margin_whitening=args.no_margin_whitening,
        margin_pad=args.margin_pad,
        debug=args.debug is not None,
        debug_log_path=args.debug or None,
    )
    if args.workers is not None:
        options_kwargs['max_workers'] = args.workers
    # Truncate/create the debug log file at startup (the pipeline appends)
    if args.debug:
        try:
            open(args.debug, 'w', encoding='utf-8').close()
        except OSError as e:
            print(f"Error: cannot write debug log {args.debug}: {e}", file=sys.stderr)
            sys.exit(1)

    options = ConversionOptions(**options_kwargs)

    # Progress callback with progress bar
    last_phase = [None]  # Track current phase for newline handling

    def progress(current: int, total: int, message: str):
        if args.quiet:
            return

        if current == 0:
            # Phase header
            if last_phase[0] is not None:
                print()  # Newline after previous progress bar
            print(message)
            last_phase[0] = message
        else:
            # Progress bar - expand to terminal width
            try:
                term_width = os.get_terminal_size().columns
            except OSError:
                term_width = 80

            count_str = f"({current}/{total})"
            fixed_chars = 2 + 1 + 4 + 1 + len(count_str)  # [] + space + "100%" + space + count
            width = max(10, term_width - fixed_chars)

            filled = int(width * current / total)
            bar = "█" * filled + "░" * (width - filled)
            pct = int(100 * current / total)
            print(f"\r[{bar}] {pct:3d}% {count_str}", end="", flush=True)
            if current == total:
                print()  # Newline when complete
                last_phase[0] = None

    # Run conversion
    try:
        start_time = datetime.now()
        result = convert_images(args.input_dir, args.output_dir, options, progress)
        elapsed = datetime.now() - start_time

        if not args.quiet:
            total_seconds = int(elapsed.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                elapsed_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                elapsed_str = f"{minutes}m {seconds}s"
            else:
                elapsed_str = f"{elapsed.total_seconds():.1f}s"

            print(f"\nConversion complete!")
            print(f"  Input:  {args.input_dir}")
            print(f"  Output: {result.output_path}")
            print(f"  Time:   {elapsed_str}")
            print(f"  Pages:  {result.processed_pages}/{result.total_pages}")
            if result.is_vertical_writing:
                print(f"  Layout: Vertical (Japanese)")
            else:
                print(f"  Layout: Horizontal")

    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
