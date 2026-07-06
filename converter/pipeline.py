"""
Main conversion pipeline - exact port of SuperPdfUtil.cs PerformPdfMainAsync + PerformPagesYohakuAsync.

This module matches the C# implementation exactly, including:
- Temp directory structure for intermediate files
- Internal high-resolution processing (4960x7016)
- Final output at 3508px height
- Odd/even page grouping
- IQR-based outlier removal
- Global color adjustment per group
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2

from .pdf_reader import extract_pages, get_page_count
from .pdf_writer import build_pdf
from .enhancer import create_enhancer, BaseEnhancer

# Import from Cython module (C# faithful port)
from .image_processing_cy import (
    # Core algorithms - Cython (fast, C# faithful)
    CalculateColorStats,
    ExcludeOutliers,
    DecideGlobalColorAdjustment,
    ApplyGlobalColorAdjustment,
    DecideGroupCropRegion,
    ResizeAndMakePaddingWithNaturalPaperColor,
    ResizeAndMakePaddingWithNaturalPaperColor2,
    DetectTextBoundingBox,
    UnifyCropRegions,
    AddMarginAndClip,  # C# lines 2682-2699 - normalizes crop region
    IsPaperVerticalWriting_GetProbability,
    # Data classes
    ColorStats,
    GlobalColorParam,
    PageBoundingBox,
    # Constants
    get_internal_high_res_width,
    get_internal_high_res_height,
    get_final_target_height,
)

# Import remaining functions from Python module (not yet ported to Cython)
from .image_processing import (
    deskew,
    detect_deskew_angle,
    apply_deskew_rotation,
    remove_show_through,
    remove_margin_background,
    detect_page_edge_junk,
)

from .ocr import (
    detect_page_number,
    find_page_number_offset,
    calculate_page_alignment_shifts,
    calculate_page_alignment_shifts_v2,
    PageNumberResult,
)

# Get constants
INTERNAL_HIGH_RES_WIDTH = get_internal_high_res_width()
INTERNAL_HIGH_RES_HEIGHT = get_internal_high_res_height()
FINAL_TARGET_HEIGHT = get_final_target_height()


@dataclass
class ConversionOptions:
    """
    Options for PDF conversion.
    Matches C# SuperPerformPdfOptions exactly.
    """
    # Margin to add around detected content area (percentage of content size)
    # C# default: 7 (SuperPdfUtil.cs line 1315)
    margin_percent: int = 7

    # Bypass first page (keeps ESRGAN, skips deskew/color/crop)
    bypass_first_page: bool = False

    # Bypass last page (keeps ESRGAN, skips deskew/color/crop)
    bypass_last_page: bool = False

    # Maximum pages to process (for debugging)
    max_pages: Optional[int] = None

    # Skip Real-ESRGAN enhancement
    skip_enhancement: bool = False

    # Enhancement model path
    model_path: Optional[str] = None

    # Upscaling factor for enhancement
    scale: int = 2

    # Tile size for enhancement (0 = no tiling)
    tile_size: int = 512

    # DPI for PDF extraction
    dpi: int = 300

    # Keep temp directory after processing (for debugging)
    keep_temp: bool = False

    # OCR language for page number detection
    ocr_lang: str = 'eng+jpn'

    # Number of parallel workers (default: number of CPUs, matching C#)
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)

    # Image format in PDF ('jpeg' or 'png'), default 'jpeg' to match C#
    pdf_image_format: str = 'jpeg'

    # JPEG quality (0-100), default 70 to match C#
    jpeg_quality: int = 70

    # Non-local means denoising strength for deskew preprocessing (0 = disabled)
    denoise_strength: int = 20

    # Maximum deskew angle to correct (degrees). Detected angles larger than this
    # are treated as detection errors and ignored (page left unrotated).
    # Note: the Radon-based detector can only measure up to ~7 degrees, so values
    # above that mainly widen the accepted range up to that physical limit.
    max_deskew_degree: float = 10.0

    # Disable deskew entirely for all pages
    no_deskew: bool = False

    # Physical page numbers (1-indexed) to skip deskew for (None = deskew all)
    deskew_exclude_pages: Optional[Set[int]] = None

    # Show-through (裏映り) removal via local flat-field + whitening runs on ALL
    # pages by default (grayscale output — intended for text-heavy books).
    # It replaces the global color adjustment on those pages: the paper
    # background is estimated locally and flattened to white, which removes
    # both show-through text and non-uniform background color.
    # Disable it entirely with no_bleed_removal, or list 1-indexed pages to
    # exclude; excluded pages keep the standard global color adjustment
    # (e.g. color/photo pages that would otherwise be desaturated).
    no_bleed_removal: bool = False
    bleed_removal_exclude_pages: Optional[Set[int]] = None
    # Tuning for show-through removal (see remove_show_through).
    bleed_bg_ksize: int = 151
    bleed_black_point: int = 115
    bleed_white_point: int = 205

    # Whiten the four outer margin bands that contain no text (see
    # remove_margin_background). A band is cleared only if it touches a page
    # edge and runs to the opposite edge without any text, so partially
    # detected text can never be erased. Runs on all pages by default,
    # after color/show-through adjustment.
    disable_margin_whitening: bool = False
    margin_pad: int = 40

    # Print per-page debug output: what was detected/decided and which
    # processing was applied on each page
    debug: bool = False

    # When set, debug output is appended to this file instead of the console
    # (the CLI truncates the file at startup)
    debug_log_path: Optional[str] = None


@dataclass
class PageInfo:
    """Information about a single page during processing."""
    file_path: str
    page_number: int  # 1-indexed
    is_odd: bool
    deskew_file_path: Optional[str] = None
    color_adj_file_path: Optional[str] = None
    color_stats: Optional[ColorStats] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    # Text extent (left, top, right, bottom) from margin whitening; the band
    # widths outside it are the cleared page-edge area of this page
    margin_extent: Optional[Tuple[int, int, int, int]] = None
    # Per-page debug records (filled when options.debug is enabled)
    debug_info: dict = field(default_factory=dict)


@dataclass
class ConversionResult:
    """Result of PDF conversion."""
    output_path: Path
    total_pages: int
    processed_pages: int
    is_vertical_writing: bool = False
    page_number_offset: Optional[int] = None
    physical_page_start: Optional[int] = None
    logical_page_start: Optional[int] = None


def convert_pdf(
    input_path: str | Path,
    output_path: str | Path,
    options: Optional[ConversionOptions] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> ConversionResult:
    """
    Convert a PDF with AI enhancement.

    Exact port of C# SuperPdfUtil.PerformPdfMainAsync + PerformPagesYohakuAsync.

    Pipeline:
    1. Extract PDF pages to images
    2. Crop 0.5% scan margins
    3. AI enhancement (Real-ESRGAN)
    4. PerformPagesYohaku:
       a. Resize to internal high-res (4960x7016)
       b. Deskew and save to temp
       c. Calculate color statistics
       d. Decide global color adjustment (per odd/even group)
       e. Apply color adjustment and detect bounding boxes
       f. Decide crop regions (with IQR outlier removal)
       g. OCR page number detection
       h. Final output with natural paper padding

    Args:
        input_path: Path to input PDF
        output_path: Path for output PDF
        options: Conversion options
        progress_callback: Callback for progress updates (current, total, message)

    Returns:
        ConversionResult with metadata
    """
    if options is None:
        options = ConversionOptions()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    def report(current: int, total: int, message: str):
        if progress_callback:
            progress_callback(current, total, message)
        else:
            if current == 0:
                print(message)
            else:
                print(f"[{current}/{total}] {message}")

    # Create temp directory structure (matching C#)
    temp_root = tempfile.mkdtemp(prefix="pdf_converter_")
    pdf_extracted_dir = os.path.join(temp_root, "1_pdf_extracted")
    pdf_cropped_dir = os.path.join(temp_root, "1_2_pdf_cropped")
    pdf_enhanced_dir = os.path.join(temp_root, "2_pdf_enhanced")
    pdf_adjusted_dir = os.path.join(temp_root, "3_pdf_adjusted")
    pdf_tmp_dir = os.path.join(temp_root, "99_tmp")

    for d in [pdf_extracted_dir, pdf_cropped_dir, pdf_enhanced_dir, pdf_adjusted_dir, pdf_tmp_dir]:
        os.makedirs(d, exist_ok=True)

    try:
        # =====================================================================
        # STEP 1: Extract PDF pages to images
        # =====================================================================
        total_pages = get_page_count(input_path)
        report(0, total_pages, f"Found {total_pages} pages")

        max_pages = options.max_pages or total_pages
        pages_to_extract = min(max_pages, total_pages)

        report(0, pages_to_extract, "Step 1: Extracting pages from PDF...")

        for idx, (page_num, image) in enumerate(extract_pages(
            input_path,
            dpi=options.dpi,
            start_page=0,
            end_page=pages_to_extract,
            grayscale=False,
        )):
            # Save as PNG
            out_path = os.path.join(pdf_extracted_dir, f"page_{page_num + 1:04d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            report(idx + 1, pages_to_extract, f"Extracting page {page_num + 1}")

        # =====================================================================
        # STEP 2: Crop 0.5% scan margins
        # =====================================================================
        report(0, pages_to_extract, "Step 2: Cropping scan margins...")

        extracted_files = sorted([
            os.path.join(pdf_extracted_dir, f)
            for f in os.listdir(pdf_extracted_dir)
            if f.endswith('.png')
        ])

        for idx, src_path in enumerate(extracted_files):
            image = cv2.imread(src_path)
            h, w = image.shape[:2]

            if w >= 10 and h >= 10:
                margin_w = int(w * 0.005)
                margin_h = int(h * 0.005)

                cropped = image[margin_h:h - margin_h, margin_w:w - margin_w]

                out_path = os.path.join(pdf_cropped_dir, os.path.basename(src_path))
                cv2.imwrite(out_path, cropped)

            report(idx + 1, pages_to_extract, f"Cropping page {idx + 1}")

        # =====================================================================
        # STEP 3: AI Enhancement (Real-ESRGAN)
        # =====================================================================
        if options.skip_enhancement or not options.model_path:
            report(0, pages_to_extract, "Step 3: Skipping enhancement...")
            # Just copy files
            for f in os.listdir(pdf_cropped_dir):
                shutil.copy(
                    os.path.join(pdf_cropped_dir, f),
                    os.path.join(pdf_enhanced_dir, f)
                )
        else:
            report(0, pages_to_extract, "Step 3: Enhancing with Real-ESRGAN...")

            enhancer = create_enhancer(
                options.model_path,
                scale=options.scale,
                tile_size=options.tile_size,
            )

            cropped_files = sorted([
                os.path.join(pdf_cropped_dir, f)
                for f in os.listdir(pdf_cropped_dir)
                if f.endswith('.png')
            ])

            for idx, src_path in enumerate(cropped_files):
                image = cv2.imread(src_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if options.tile_size > 0:
                    enhanced = enhancer.enhance_tiled(image_rgb)
                else:
                    enhanced = enhancer.enhance(image_rgb)

                out_path = os.path.join(pdf_enhanced_dir, os.path.basename(src_path))
                cv2.imwrite(out_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

                report(idx + 1, pages_to_extract, f"Enhancing page {idx + 1}")

        # =====================================================================
        # STEP 4: PerformPagesYohaku (main processing)
        # =====================================================================
        report(0, pages_to_extract, "Step 4: Processing pages (deskew, color, crop, OCR)...")

        result = _perform_pages_yohaku(
            src_dir=pdf_enhanced_dir,
            dst_dir=pdf_adjusted_dir,
            tmp_dir=pdf_tmp_dir,
            options=options,
            progress_callback=report,
        )

        # =====================================================================
        # STEP 5: Build final PDF
        # =====================================================================
        report(0, pages_to_extract, "Step 5: Building final PDF...")

        # Get sorted output files
        output_files = sorted([
            os.path.join(pdf_adjusted_dir, f)
            for f in os.listdir(pdf_adjusted_dir)
            if f.endswith('.png')
        ])

        def load_final_images():
            for path in output_files:
                img = cv2.imread(path)
                yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate DPI for output
        # C# always uses 300 DPI for embedding images into PDF
        # regardless of whether enhancement was used.
        # Final images are always scaled to FINAL_TARGET_HEIGHT (3508),
        # so 300 DPI gives correct page size (3508/300*72 = 841.92 pts)
        output_dpi = 300

        build_pdf(
            load_final_images(),
            output_path,
            dpi=output_dpi,
            title=input_path.stem,
            image_format=options.pdf_image_format,
            jpeg_quality=options.jpeg_quality,
            physical_page_start=result.get('physical_page_start'),
            logical_page_start=result.get('logical_page_start'),
        )

        report(pages_to_extract, pages_to_extract, f"Done! Saved to {output_path}")

        return ConversionResult(
            output_path=output_path,
            total_pages=total_pages,
            processed_pages=len(output_files),
            is_vertical_writing=result.get('is_vertical', False),
            page_number_offset=result.get('page_offset'),
            physical_page_start=result.get('physical_page_start'),
            logical_page_start=result.get('logical_page_start'),
        )

    finally:
        # Clean up temp directory
        if not options.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
        else:
            print(f"Temp directory kept at: {temp_root}")


def convert_images(
    input_dir: str | Path,
    output_dir: str | Path,
    options: Optional[ConversionOptions] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> ConversionResult:
    """
    Convert a folder of scanned JPEG pages with the same processing pipeline
    as convert_pdf (crop, enhancement, deskew, color/show-through, margin
    whitening, crop unification, OCR alignment), writing one processed JPEG
    per input file under the same name.

    Input files are expected to be numbered like 000.JPG, 001.JPG, ... in
    ascending order. Gaps in the numbering are allowed: the file number is
    used as the physical page number, so the odd/even (left/right page)
    grouping stays aligned across gaps.

    Args:
        input_dir: Folder containing the input JPEG files
        output_dir: Folder to write the processed JPEG files to (created if
                    missing); each output keeps its input's file name
        options: Conversion options (pdf_image_format is ignored;
                 jpeg_quality controls the output JPEG quality)
        progress_callback: Callback for progress updates (current, total, message)

    Returns:
        ConversionResult with metadata (output_path is the output folder)
    """
    if options is None:
        options = ConversionOptions()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    def report(current: int, total: int, message: str):
        if progress_callback:
            progress_callback(current, total, message)
        else:
            if current == 0:
                print(message)
            else:
                print(f"[{current}/{total}] {message}")

    # Collect input JPEGs (ascending file-name order)
    src_names = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg')
    )
    if not src_names:
        raise FileNotFoundError(f"No JPEG files found in: {input_dir}")

    if options.max_pages:
        src_names = src_names[:options.max_pages]
    total_pages = len(src_names)

    # Page numbers from the numeric file stems (000.JPG -> page 1), so gaps
    # keep the odd/even alternation. Falls back to sequential numbering when
    # any stem is not a plain number.
    stems = [os.path.splitext(f)[0] for f in src_names]
    if all(s.isdigit() for s in stems):
        page_numbers = [int(s) + 1 for s in stems]
    else:
        page_numbers = list(range(1, total_pages + 1))

    report(0, total_pages, f"Found {total_pages} JPEG files")

    # Create temp directory structure (mirrors convert_pdf)
    temp_root = tempfile.mkdtemp(prefix="jpeg_converter_")
    img_cropped_dir = os.path.join(temp_root, "1_2_img_cropped")
    img_enhanced_dir = os.path.join(temp_root, "2_img_enhanced")
    img_adjusted_dir = os.path.join(temp_root, "3_img_adjusted")
    img_tmp_dir = os.path.join(temp_root, "99_tmp")
    for d in [img_cropped_dir, img_enhanced_dir, img_adjusted_dir, img_tmp_dir]:
        os.makedirs(d, exist_ok=True)

    try:
        # =====================================================================
        # STEP 1: Load JPEGs and crop 0.5% scan margins
        # =====================================================================
        report(0, total_pages, "Step 1: Loading and cropping scan margins...")

        for idx, (name, num) in enumerate(zip(src_names, page_numbers)):
            image = cv2.imread(str(input_dir / name))
            if image is None:
                raise IOError(f"Cannot read image: {input_dir / name}")
            h, w = image.shape[:2]

            if w >= 10 and h >= 10:
                margin_w = int(w * 0.005)
                margin_h = int(h * 0.005)
                image = image[margin_h:h - margin_h, margin_w:w - margin_w]

            out_path = os.path.join(img_cropped_dir, f"page_{num:04d}.png")
            cv2.imwrite(out_path, image)
            report(idx + 1, total_pages, f"Cropping {name}")

        # =====================================================================
        # STEP 2: AI Enhancement (Real-ESRGAN)
        # =====================================================================
        if options.skip_enhancement or not options.model_path:
            report(0, total_pages, "Step 2: Skipping enhancement...")
            for f in os.listdir(img_cropped_dir):
                shutil.copy(
                    os.path.join(img_cropped_dir, f),
                    os.path.join(img_enhanced_dir, f)
                )
        else:
            report(0, total_pages, "Step 2: Enhancing with Real-ESRGAN...")

            enhancer = create_enhancer(
                options.model_path,
                scale=options.scale,
                tile_size=options.tile_size,
            )

            cropped_files = sorted(
                os.path.join(img_cropped_dir, f)
                for f in os.listdir(img_cropped_dir)
                if f.endswith('.png')
            )
            for idx, src_path in enumerate(cropped_files):
                image = cv2.imread(src_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if options.tile_size > 0:
                    enhanced = enhancer.enhance_tiled(image_rgb)
                else:
                    enhanced = enhancer.enhance(image_rgb)

                out_path = os.path.join(img_enhanced_dir, os.path.basename(src_path))
                cv2.imwrite(out_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
                report(idx + 1, total_pages, f"Enhancing page {idx + 1}")

        # =====================================================================
        # STEP 3: PerformPagesYohaku (main processing)
        # =====================================================================
        report(0, total_pages, "Step 3: Processing pages (deskew, color, crop, OCR)...")

        result = _perform_pages_yohaku(
            src_dir=img_enhanced_dir,
            dst_dir=img_adjusted_dir,
            tmp_dir=img_tmp_dir,
            options=options,
            progress_callback=report,
            page_numbers=page_numbers,
        )

        # =====================================================================
        # STEP 4: Save processed pages as JPEGs under the original names
        # =====================================================================
        report(0, total_pages, "Step 4: Writing output JPEGs...")

        output_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for idx, (name, num) in enumerate(zip(src_names, page_numbers)):
            adjusted_path = os.path.join(img_adjusted_dir, f"page_{num:04d}.png")
            if not os.path.exists(adjusted_path):
                report(idx + 1, total_pages, f"WARNING: no output for {name}")
                continue
            image = cv2.imread(adjusted_path)
            cv2.imwrite(
                str(output_dir / name), image,
                [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality],
            )
            written += 1
            report(idx + 1, total_pages, f"Writing {name}")

        report(total_pages, total_pages, f"Done! Saved {written} files to {output_dir}")

        return ConversionResult(
            output_path=output_dir,
            total_pages=total_pages,
            processed_pages=written,
            is_vertical_writing=result.get('is_vertical', False),
            page_number_offset=result.get('page_offset'),
            physical_page_start=result.get('physical_page_start'),
            logical_page_start=result.get('logical_page_start'),
        )

    finally:
        if not options.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
        else:
            print(f"Temp directory kept at: {temp_root}")


def _perform_pages_yohaku(
    src_dir: str,
    dst_dir: str,
    tmp_dir: str,
    options: ConversionOptions,
    progress_callback: Callable[[int, int, str], None],
    page_numbers: Optional[List[int]] = None,
) -> dict:
    """
    Port of C# PerformPagesYohakuAsync.

    Performs:
    - Resize to internal high-res (4960x7016) with natural paper padding
    - Deskew
    - Color statistics calculation
    - Global color adjustment
    - Bounding box detection
    - Crop region decision
    - OCR page number detection
    - Final output

    Args:
        page_numbers: Optional 1-indexed page numbers matching the sorted
                      source files. Used by the JPEG converter so that gaps in
                      the input numbering keep the odd/even (left/right page)
                      alternation intact. Defaults to sequential numbering.
    """
    def report(current: int, total: int, message: str):
        progress_callback(current, total, message)

    def dbg(message: str):
        if not options.debug:
            return
        line = f"[DEBUG] {message}"
        if options.debug_log_path:
            with open(options.debug_log_path, 'a', encoding='utf-8') as f:
                f.write(line + "\n")
        else:
            print(line, flush=True)

    def dbg_pages(phase: str):
        """Emit collected per-page debug records for a phase, in page order."""
        if not options.debug:
            return
        for page in sorted(page_infos, key=lambda p: p.page_number):
            info = page.debug_info.get(phase)
            if info:
                name = os.path.basename(page.file_path)
                group = 'odd' if page.is_odd else 'even'
                dbg(f"page {page.page_number:3d} ({name}, {group}): {info}")

    # Clean tmp_dir
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))

    # Get input files
    src_files = sorted([
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.lower().endswith(('.png', '.bmp'))
    ])

    if options.max_pages:
        src_files = src_files[:options.max_pages]
        if page_numbers is not None:
            page_numbers = page_numbers[:options.max_pages]

    total_pages = len(src_files)
    if total_pages == 0:
        return {}

    # Create page info list
    page_infos: List[PageInfo] = []
    for idx, file_path in enumerate(src_files):
        num = page_numbers[idx] if page_numbers is not None else idx + 1
        page_infos.append(PageInfo(
            file_path=file_path,
            page_number=num,
            is_odd=num % 2 == 1,
        ))

    # =========================================================================
    # Phase 1: Resize to internal high-res, deskew, calculate color stats
    # =========================================================================
    report(0, total_pages, "Phase 4.1: Deskewing and calculating color statistics...")

    odd_color_stats: List[ColorStats] = []
    even_color_stats: List[ColorStats] = []

    def process_phase1(page: PageInfo) -> Tuple[PageInfo, ColorStats]:
        """Process a single page for Phase 1 (resize, deskew, color stats)."""
        # Load image
        img_bgr = cv2.imread(page.file_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to internal high-res with natural paper padding (Cython with nogil)
        resized = ResizeAndMakePaddingWithNaturalPaperColor(
            img_rgb,
            INTERNAL_HIGH_RES_WIDTH,
            INTERNAL_HIGH_RES_HEIGHT,
        )

        # Deskew (Python - uses OpenCV which releases GIL internally)
        # Use original extracted image for angle detection (lower res, faster),
        # then apply rotation to the resized high-res image.
        # Skip when deskew is disabled globally or this page is excluded.
        exclude_pages = options.deskew_exclude_pages or set()
        if options.no_deskew:
            deskewed = resized
            page.debug_info['deskew'] = 'deskew SKIPPED (--no-deskew)'
        elif page.page_number in exclude_pages:
            deskewed = resized
            page.debug_info['deskew'] = 'deskew SKIPPED (excluded page)'
        else:
            raw = []
            angle = detect_deskew_angle(
                img_rgb,
                max_degree=options.max_deskew_degree,
                denoise_strength=options.denoise_strength,
                raw_out=raw,
            )
            deskewed = apply_deskew_rotation(resized, angle)
            raw_s = f'{raw[0]:+.3f}' if raw else 'n/a'
            if abs(angle) < 0.001:
                if raw and abs(raw[0]) > options.max_deskew_degree:
                    why = f'exceeds max {options.max_deskew_degree}'
                else:
                    why = 'near zero'
                page.debug_info['deskew'] = (
                    f'deskew NOT applied (raw={raw_s} deg, {why})')
            else:
                page.debug_info['deskew'] = (
                    f'deskew rotated {angle:+.3f} deg (raw={raw_s} deg)')

        # Save deskewed image to tmp
        deskew_path = os.path.join(tmp_dir, f"deskew_{page.page_number:04d}.png")
        cv2.imwrite(deskew_path, cv2.cvtColor(deskewed, cv2.COLOR_RGB2BGR))
        page.deskew_file_path = deskew_path

        # Calculate color statistics (Cython with nogil)
        stats = CalculateColorStats(deskewed)
        page.color_stats = stats
        if options.debug:
            page.debug_info['deskew'] += (
                f' | paper=({stats.PaperR:.0f},{stats.PaperG:.0f},{stats.PaperB:.0f})'
                f' ink=({stats.InkR:.0f},{stats.InkG:.0f},{stats.InkB:.0f})')

        return page, stats

    # Run Phase 1 in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
        futures = {executor.submit(process_phase1, page): page for page in page_infos}
        for future in as_completed(futures):
            page, stats = future.result()
            completed += 1
            report(completed, total_pages, f"Processing page {page.page_number}")

            if page.is_odd:
                odd_color_stats.append(stats)
            else:
                even_color_stats.append(stats)

    dbg_pages('deskew')

    # =========================================================================
    # Phase 2: Decide global color adjustment parameters
    # =========================================================================
    report(0, total_pages, "Phase 4.2: Calculating color adjustment parameters...")

    # C# lines 1675-1679: ExcludeOutliers removes top/bottom 20% by MeanR
    filtered_odd_stats = ExcludeOutliers(odd_color_stats)
    filtered_even_stats = ExcludeOutliers(even_color_stats)

    odd_color_param = DecideGlobalColorAdjustment(filtered_odd_stats)
    even_color_param = DecideGlobalColorAdjustment(filtered_even_stats)

    if options.debug:
        for label, stats_all, stats_used, p in (
            ('odd', odd_color_stats, filtered_odd_stats, odd_color_param),
            ('even', even_color_stats, filtered_even_stats, even_color_param),
        ):
            dbg(f"group {label:4}: {len(stats_used)}/{len(stats_all)} pages used "
                f"(outliers excluded) | scale=({p.ScaleR:.3f},{p.ScaleG:.3f},{p.ScaleB:.3f}) "
                f"offset=({p.OffsetR:+.1f},{p.OffsetG:+.1f},{p.OffsetB:+.1f}) | "
                f"paper=({p.PaperR},{p.PaperG},{p.PaperB}) "
                f"ghost_lum_threshold={p.GhostSuppressLumThreshold}")

    # =========================================================================
    # Phase 3: Apply color adjustment and detect bounding boxes
    # =========================================================================
    report(0, total_pages, "Phase 4.3: Applying color adjustment...")

    odd_bboxes: List[PageBoundingBox] = []
    even_bboxes: List[PageBoundingBox] = []

    def process_phase3(page: PageInfo) -> Tuple[PageInfo, Optional[PageBoundingBox]]:
        """Process a single page for Phase 3 (color adjustment, bounding box)."""
        # Load deskewed image
        img_bgr = cv2.imread(page.deskew_file_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect scan junk (photography stand, page-edge shadow lines) on the
        # PRE-adjustment image where those are still solid dark shapes - the
        # show-through flat-field would turn them into text-like texture.
        edge_junk = None
        if not options.disable_margin_whitening:
            edge_junk = detect_page_edge_junk(img_rgb)

        # Show-through removal runs on all pages by default; skip if globally
        # disabled or this page is excluded (excluded pages keep the standard
        # global color adjustment).
        bleed_exclude = options.bleed_removal_exclude_pages or set()
        apply_bleed = not options.no_bleed_removal and page.page_number not in bleed_exclude
        if apply_bleed:
            adjusted = remove_show_through(
                img_rgb,
                bg_ksize=options.bleed_bg_ksize,
                black_point=options.bleed_black_point,
                white_point=options.bleed_white_point,
            )
            adjusted = np.ascontiguousarray(adjusted)
            color_dbg = (f'show-through removal applied (ksize={options.bleed_bg_ksize}, '
                         f'black={options.bleed_black_point}, white={options.bleed_white_point}, '
                         f'grayscale output)')
        else:
            # Apply color adjustment (Cython with nogil - modifies in-place)
            color_param = odd_color_param if page.is_odd else even_color_param
            adjusted = np.ascontiguousarray(img_rgb)
            ApplyGlobalColorAdjustment(adjusted, color_param)
            why = '--no-bleed-removal' if options.no_bleed_removal else 'excluded page'
            color_dbg = f'show-through SKIPPED ({why}) -> global color adjustment (color kept)'

        # Whiten the text-free outer margin bands (before bbox so the crop
        # isn't pulled toward edge junk / spine shadows). The returned text
        # extent records how much page-edge area was cleared on each side;
        # Phase 4 re-grants margins of the same size around the crop.
        if not options.disable_margin_whitening:
            whitened, extent = remove_margin_background(
                adjusted,
                margin_pad=options.margin_pad,
                junk_mask=edge_junk,
            )
            adjusted = np.ascontiguousarray(whitened)
            page.margin_extent = extent
            junk_px = int(edge_junk.sum()) if edge_junk is not None else 0
            if extent is None:
                margin_dbg = 'margin whitening: no text detected -> left unchanged'
            else:
                l, t, r, b = extent
                h_img, w_img = adjusted.shape[:2]
                margin_dbg = (f'margins cleared L={l} T={t} '
                              f'R={w_img - r} B={h_img - b} px'
                              f' (edge junk masked={junk_px:,} px)')
        else:
            margin_dbg = 'margin whitening SKIPPED (--no-margin-whitening)'

        # Save color-adjusted image
        color_adj_path = os.path.join(tmp_dir, f"coloradj_{page.page_number:04d}.png")
        cv2.imwrite(color_adj_path, cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR))
        page.color_adj_file_path = color_adj_path

        # Detect bounding box (Cython - uses OpenCV which releases GIL internally)
        bbox = DetectTextBoundingBox(adjusted)
        page.bounding_box = bbox
        if options.debug:
            page.debug_info['color'] = (
                f'{color_dbg} | {margin_dbg} | '
                f'bbox=(x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})')

        # C# does NOT skip any pages from crop calculation
        # All pages with valid bounding boxes are included
        page_bbox = None
        if bbox[2] > 0 and bbox[3] > 0:
            # Cython PageBoundingBox takes (page_number, left, top, width, height)
            page_bbox = PageBoundingBox(
                page_number=page.page_number,
                left=bbox[0],
                top=bbox[1],
                width=bbox[2],
                height=bbox[3],
            )

        return page, page_bbox

    # Run Phase 3 in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
        futures = {executor.submit(process_phase3, page): page for page in page_infos}
        for future in as_completed(futures):
            page, page_bbox = future.result()
            completed += 1
            report(completed, total_pages, f"Adjusting page {page.page_number}")

            if page_bbox is not None:
                if page.is_odd:
                    odd_bboxes.append(page_bbox)
                else:
                    even_bboxes.append(page_bbox)

    dbg_pages('color')

    # =========================================================================
    # Phase 4: Decide crop regions
    # =========================================================================
    report(0, total_pages, "Phase 4.4: Deciding crop regions...")

    # Cython DecideGroupCropRegion returns (left, top, width, height) tuple
    odd_crop_raw = DecideGroupCropRegion(odd_bboxes)
    even_crop_raw = DecideGroupCropRegion(even_bboxes)

    # With margin whitening active, the crop shrinks to the bare text, so the
    # output would lose its margins entirely. Re-grant margins equal to the
    # median cleared page-edge band on each side (the area between the text
    # and the page edge in the pre-whitening image). The same four margins are
    # applied to both groups, so odd and even pages share the same top margin
    # and their text starts at the same height.
    margin_extents = [p.margin_extent for p in page_infos if p.margin_extent]
    use_band_margins = not options.disable_margin_whitening and len(margin_extents) > 0

    # Unify crop regions (Cython); with band margins the percent margin is
    # replaced by the band-based expansion below
    odd_crop, even_crop = UnifyCropRegions(
        odd_crop_raw, even_crop_raw,
        0 if use_band_margins else options.margin_percent,
        INTERNAL_HIGH_RES_WIDTH,
        INTERNAL_HIGH_RES_HEIGHT,
    )

    if use_band_margins:
        W, H = INTERNAL_HIGH_RES_WIDTH, INTERNAL_HIGH_RES_HEIGHT
        band_l = int(np.median([e[0] for e in margin_extents]))
        band_t = int(np.median([e[1] for e in margin_extents]))
        band_r = int(np.median([W - e[2] for e in margin_extents]))
        band_b = int(np.median([H - e[3] for e in margin_extents]))

        # Clamp jointly so both crops stay inside the image with identical
        # dimensions (unequal clamping would give the groups different scales)
        m_l = min(band_l, odd_crop[0], even_crop[0])
        m_t = min(band_t, odd_crop[1], even_crop[1])
        m_r = min(band_r,
                  W - (odd_crop[0] + odd_crop[2]),
                  W - (even_crop[0] + even_crop[2]))
        m_b = min(band_b,
                  H - (odd_crop[1] + odd_crop[3]),
                  H - (even_crop[1] + even_crop[3]))

        odd_crop = (odd_crop[0] - m_l, odd_crop[1] - m_t,
                    odd_crop[2] + m_l + m_r, odd_crop[3] + m_t + m_b)
        even_crop = (even_crop[0] - m_l, even_crop[1] - m_t,
                     even_crop[2] + m_l + m_r, even_crop[3] + m_t + m_b)

        dbg(f"crop margins: median cleared bands L={band_l} T={band_t} R={band_r} B={band_b} px; "
            f"re-granted (after joint clamping) L={m_l} T={m_t} R={m_r} B={m_b} px "
            f"(shared by odd/even -> same text start height)")

    # Calculate final dimensions
    crop_width = max(odd_crop[2], even_crop[2])
    crop_height = max(odd_crop[3], even_crop[3])

    final_height = FINAL_TARGET_HEIGHT
    final_width = crop_width * final_height // crop_height if crop_height > 0 else crop_width

    dbg(f"crop odd : raw={tuple(odd_crop_raw)} -> final=(x={odd_crop[0]}, y={odd_crop[1]}, "
        f"w={odd_crop[2]}, h={odd_crop[3]})")
    dbg(f"crop even: raw={tuple(even_crop_raw)} -> final=(x={even_crop[0]}, y={even_crop[1]}, "
        f"w={even_crop[2]}, h={even_crop[3]})")
    dbg(f"output size: {final_width}x{final_height} px "
        f"({'band margins' if use_band_margins else f'margin_percent={options.margin_percent}%'})")


    # =========================================================================
    # Phase 5: OCR page number detection
    # =========================================================================
    report(0, total_pages, "Phase 4.5: Detecting page numbers...")

    # Load all color-adjusted images for OCR
    ocr_images = []
    for idx, page in enumerate(page_infos):
        img_bgr = cv2.imread(page.color_adj_file_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ocr_images.append(img_rgb)

    # C# lines 3229-3620: Sophisticated cross-page validation for page numbers
    # This collects ALL number candidates per page, then uses cross-validation
    # to identify which numbers are actual page numbers (must increment sequentially)
    ocr_results = calculate_page_alignment_shifts_v2(
        ocr_images,
        INTERNAL_HIGH_RES_WIDTH,
        INTERNAL_HIGH_RES_HEIGHT,
        progress_callback=report,
    )

    # Find page number offset
    page_offset = find_page_number_offset(ocr_results)

    # Determine physical/logical page shift
    physical_page_start = None
    logical_page_start = None
    for idx, ocr_result in enumerate(ocr_results):
        if ocr_result.detected_number is not None and ocr_result.detected_number >= 1:
            physical_page = idx + 1
            logical_page = ocr_result.detected_number
            if physical_page != logical_page:
                physical_page_start = physical_page
                logical_page_start = logical_page
                break

    if options.debug:
        for idx, page in enumerate(page_infos):
            r = ocr_results[idx] if idx < len(ocr_results) else None
            if r is None:
                dbg(f"page {page.page_number:3d}: OCR no result")
            elif r.detected_number is not None:
                dbg(f"page {page.page_number:3d}: OCR page number={r.detected_number}, "
                    f"alignment shift=({r.shift_x:+d},{r.shift_y:+d}) px")
            else:
                dbg(f"page {page.page_number:3d}: OCR page number not detected, "
                    f"alignment shift=({r.shift_x:+d},{r.shift_y:+d}) px")
        dbg(f"page number offset={page_offset}, physical/logical start="
            f"{physical_page_start}/{logical_page_start}")

    # =========================================================================
    # Phase 6: Final output
    # =========================================================================
    report(0, total_pages, "Phase 4.6: Generating final output...")

    vertical_probs: List[float] = [0.0] * total_pages  # Pre-allocate to preserve order

    def process_phase6(idx: int, page: PageInfo) -> Tuple[int, float]:
        """Process a single page for Phase 6 (final output with cropping and padding)."""
        # Check if bypassing this page
        is_front = page.page_number == 1
        is_back = page.page_number == total_pages

        if (options.bypass_first_page and is_front) or (options.bypass_last_page and is_back):
            # Bypass: use ESRGAN-enhanced image, skip deskew/color/crop
            # page.file_path points to the enhanced image from pdf_enhanced_dir
            img_bgr = cv2.imread(page.file_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Scale to FINAL_TARGET_HEIGHT (3508) while keeping original aspect ratio
            # This matches C#'s approach of scaling to standard height
            src_h, src_w = img_rgb.shape[:2]
            page_scale = final_height / src_h
            bypass_width = int(round(src_w * page_scale))
            bypass_height = final_height
            final_img = cv2.resize(img_rgb, (bypass_width, bypass_height), interpolation=cv2.INTER_LANCZOS4)
            if options.debug:
                page.debug_info['final'] = (
                    f'BYPASS (cover page): enhanced image scaled x{page_scale:.3f} '
                    f'to {bypass_width}x{bypass_height} (no deskew/color/crop)')
        else:
            # Load color-adjusted image
            img_bgr = cv2.imread(page.color_adj_file_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Get crop region for this page
            crop_region = odd_crop if page.is_odd else even_crop

            # C# line 1912: Normalize crop region with AddMarginAndClip
            # This clamps to image bounds and uses right-left+1 for width
            actual_crop = AddMarginAndClip(
                crop_region,
                0,  # marginPixel = 0
                img_rgb.shape[1],  # colorAdjustedImg.Width
                img_rgb.shape[0],  # colorAdjustedImg.Height
            )
            cx, cy, cw, ch = actual_crop

            # C# lines 1923-1929: Use ResizeAndMakePaddingWithNaturalPaperColor2
            # with scale = finalWidth / actualCrop.Width
            page_scale = float(final_width) / float(cw) if cw > 0 else 1.0

            # Get OCR-based shift values for page alignment (C# lines 1926-1927)
            # shift_x/shift_y align pages based on detected page number positions
            # Now using the sophisticated C# algorithm with cross-page validation
            ocr_result = ocr_results[idx] if idx < len(ocr_results) else None
            shift_x = ocr_result.shift_x if ocr_result else 0
            shift_y = ocr_result.shift_y if ocr_result else 0

            # Resize with natural paper padding + cropping (Cython with nogil)
            # C# line 1926: -actualCrop.Left + shift
            final_img = ResizeAndMakePaddingWithNaturalPaperColor2(
                img_rgb,
                final_width,
                final_height,
                x=-cx + shift_x,  # Negative of crop left + OCR shift
                y=-cy + shift_y,  # Negative of crop top + OCR shift
                scale=page_scale,
            )
            if options.debug:
                page.debug_info['final'] = (
                    f'crop {"odd" if page.is_odd else "even"}=(x={cx}, y={cy}, w={cw}, h={ch}), '
                    f'OCR shift=({shift_x:+d},{shift_y:+d}), scale=x{page_scale:.4f} '
                    f'-> {final_width}x{final_height}')

        # Detect vertical writing probability (Cython - C# lines 4693-4722)
        # Use Otsu binarized grayscale image (OpenCV releases GIL internally)
        gray = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v_prob = IsPaperVerticalWriting_GetProbability(binary)
        if options.debug and 'final' in page.debug_info:
            page.debug_info['final'] += f' | vertical writing prob={v_prob:.2f}'

        # Save final image
        out_path = os.path.join(dst_dir, f"page_{page.page_number:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

        return idx, v_prob

    # Run Phase 6 in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
        futures = {executor.submit(process_phase6, idx, page): page for idx, page in enumerate(page_infos)}
        for future in as_completed(futures):
            idx, v_prob = future.result()
            vertical_probs[idx] = v_prob
            completed += 1
            report(completed, total_pages, f"Finalizing page {page_infos[idx].page_number}")

    dbg_pages('final')

    # C# lines 3325-3331: Average probability and threshold at 0.5
    avg_vertical_prob = sum(vertical_probs) / len(vertical_probs) if vertical_probs else 0.0
    is_vertical = avg_vertical_prob >= 0.5

    dbg(f"layout decision: avg vertical writing prob={avg_vertical_prob:.2f} "
        f"-> {'vertical (Japanese)' if is_vertical else 'horizontal'}")

    return {
        'is_vertical': is_vertical,
        'page_offset': page_offset,
        'physical_page_start': physical_page_start,
        'logical_page_start': logical_page_start,
    }
