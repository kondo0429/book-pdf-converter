#!/usr/bin/env python3
"""
PDF Converter Batch CLI - Batch convert PDF documents in a directory.

Recursively finds all PDFs in the input directory and converts them,
preserving the folder structure in the output directory.

Usage:
    pdf-converter-batch input_dir/ output_dir/
    pdf-converter-batch input_dir/ output_dir/ --skip-existing
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

from . import convert_pdf, ConversionOptions
from .cli import parse_page_ranges, reconfigure_stdio_utf8
from .models import find_model


def find_pdfs(input_dir: Path) -> list[Path]:
    """Recursively find all PDF files in directory."""
    pdfs = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(Path(root) / f)
    return sorted(pdfs)


def main():
    reconfigure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description="Batch convert PDF documents in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdf-converter-batch input_dir/ output_dir/
  pdf-converter-batch input_dir/ output_dir/ --skip-existing
  pdf-converter-batch input_dir/ output_dir/ --skip-enhancement
        """,
    )

    # Input/Output directories
    parser.add_argument('input_dir', type=str, help='Input directory containing PDFs')
    parser.add_argument('output_dir', type=str, help='Output directory for converted PDFs')

    # Batch options
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip PDFs that already have output files')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue processing if a PDF fails')

    # Enhancement options
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to enhancement model (.mlpackage for Mac, .pth for Linux/Windows)')
    parser.add_argument('--scale', '-s', type=int, default=2,
                        help='Upscaling factor (default: 2)')
    parser.add_argument('--tile', '-t', type=int, default=512,
                        help='Tile size for enhancement (default: 512)')
    parser.add_argument('--skip-enhancement', action='store_true',
                        help='Skip AI enhancement')

    # DPI settings
    parser.add_argument('--dpi', type=int, default=300,
                        help='Input DPI for PDF rendering (default: 300)')

    # Margin settings
    parser.add_argument('--margin-percent', type=int, default=7,
                        help='Output margin percentage (default: 7)')

    # Bypass options
    parser.add_argument('--bypass-first', action='store_true',
                        help='Bypass first page processing')
    parser.add_argument('--bypass-last', action='store_true',
                        help='Bypass last page processing')

    # Deskew preprocessing
    parser.add_argument('--denoise-strength', type=int, default=20,
                        help='Non-local means denoising strength for deskew (default: 20)')
    parser.add_argument('--max-deskew-degree', type=float, default=10.0,
                        help='Max deskew angle to correct in degrees; larger detections are ignored (default: 10)')
    parser.add_argument('--no-deskew', action='store_true',
                        help='Disable deskew for all pages')
    parser.add_argument('--deskew-exclude-pages', type=str, default=None,
                        help='Page numbers (1-indexed) to skip deskew, e.g. "1,4,7-9"')

    # Show-through (bleed-through) removal - on for all pages by default (grayscale output)
    parser.add_argument('--no-bleed-removal', action='store_true',
                        help='Disable show-through/background removal for all pages')
    parser.add_argument('--bleed-removal-exclude-pages', type=str, default=None,
                        help='Page numbers (1-indexed) to skip show-through removal, e.g. "1,4,7-9" '
                             '(applies to every PDF in the batch)')
    parser.add_argument('--bleed-bg-ksize', type=int, default=151,
                        help='Show-through removal: background estimation kernel size (default: 151)')
    parser.add_argument('--bleed-black-point', type=int, default=115,
                        help='Show-through removal: values <= this become ink/black (default: 115)')
    parser.add_argument('--bleed-white-point', type=int, default=205,
                        help='Show-through removal: values >= this become paper/white; lower removes more (default: 205)')

    # Margin whitening - clear the text-free outer margin bands (on by default).
    # A band is cleared only if it touches a page edge and runs to the opposite
    # edge without any text, so text can never be erased.
    parser.add_argument('--no-margin-whitening', action='store_true',
                        help='Disable whitening of the text-free outer margin bands')
    parser.add_argument('--margin-pad', type=int, default=40,
                        help='Margin whitening: pixels kept around the detected text extent (default: 40)')

    # OCR settings
    parser.add_argument('--ocr-lang', type=str, default='eng+jpn',
                        help='Tesseract language codes (default: eng+jpn)')

    # PDF output options
    parser.add_argument('--pdf-format', type=str, default='jpeg', choices=['jpeg', 'png'],
                        help='Image format in PDF (default: jpeg)')
    parser.add_argument('--jpeg-quality', type=int, default=70,
                        help='JPEG quality 0-100 (default: 70)')

    # Debug options
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Maximum pages to process per PDF (for testing)')
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

    # Validate input directory
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    pdfs = find_pdfs(input_dir)
    if not pdfs:
        print(f"No PDF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF file(s) in {input_dir}")

    # Auto-detect model if not specified
    if not args.skip_enhancement and not args.model:
        import platform
        prefer_coreml = platform.system() == "Darwin"
        detected_model = find_model(prefer_coreml=prefer_coreml)
        if detected_model:
            args.model = str(detected_model)
            print(f"Using bundled model: {detected_model.name}")
        else:
            print("Warning: No model found. Use --model or --skip-enhancement", file=sys.stderr)
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
        dpi=args.dpi,
        keep_temp=args.keep_temp,
        ocr_lang=args.ocr_lang,
        pdf_image_format=args.pdf_format,
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

    # Process PDFs
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_pdfs = []

    for idx, pdf_path in enumerate(pdfs, 1):
        # Calculate relative path and output path
        rel_path = pdf_path.relative_to(input_dir)
        out_path = output_dir / rel_path

        # Create output subdirectory if needed
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if output exists
        if args.skip_existing and out_path.exists():
            if not args.quiet:
                print(f"[{idx}/{len(pdfs)}] Skipping (exists): {rel_path}")
            skip_count += 1
            continue

        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(pdfs)}] Processing: {rel_path}")
            print(f"{'='*60}")

        # Progress callback
        last_phase = [None]

        def progress(current: int, total: int, message: str):
            if args.quiet:
                return
            if current == 0:
                if last_phase[0] is not None:
                    print()
                print(f"  {message}")
                last_phase[0] = message
            else:
                try:
                    term_width = os.get_terminal_size().columns
                except OSError:
                    term_width = 80
                count_str = f"({current}/{total})"
                fixed_chars = 4 + 2 + 1 + 4 + 1 + len(count_str)  # indent + [] + space + "100%" + space + count
                width = max(10, term_width - fixed_chars)
                filled = int(width * current / total)
                bar = "█" * filled + "░" * (width - filled)
                pct = int(100 * current / total)
                print(f"\r  [{bar}] {pct:3d}% {count_str}", end="", flush=True)
                if current == total:
                    print()
                    last_phase[0] = None

        # Convert
        try:
            start_time = datetime.now()
            result = convert_pdf(str(pdf_path), str(out_path), options, progress)
            elapsed = datetime.now() - start_time

            if not args.quiet:
                print(f"  Completed in {elapsed.total_seconds():.1f}s")
                print(f"  Pages: {result.processed_pages}/{result.total_pages}")

            success_count += 1

        except KeyboardInterrupt:
            print("\n\nCancelled by user.", file=sys.stderr)
            sys.exit(1)

        except Exception as e:
            fail_count += 1
            failed_pdfs.append((rel_path, str(e)))

            if not args.quiet:
                print(f"  Error: {e}", file=sys.stderr)

            if not args.continue_on_error:
                print(f"\nStopping due to error. Use --continue-on-error to continue.", file=sys.stderr)
                sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print("Batch conversion complete!")
    print(f"{'='*60}")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")

    if failed_pdfs:
        print(f"\nFailed files:")
        for rel_path, error in failed_pdfs:
            print(f"  - {rel_path}: {error}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
