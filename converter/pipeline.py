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
from typing import Optional, Callable, List, Tuple
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


def _perform_pages_yohaku(
    src_dir: str,
    dst_dir: str,
    tmp_dir: str,
    options: ConversionOptions,
    progress_callback: Callable[[int, int, str], None],
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
    """
    def report(current: int, total: int, message: str):
        progress_callback(current, total, message)

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

    total_pages = len(src_files)
    if total_pages == 0:
        return {}

    # Create page info list
    page_infos: List[PageInfo] = []
    for idx, file_path in enumerate(src_files):
        page_infos.append(PageInfo(
            file_path=file_path,
            page_number=idx + 1,
            is_odd=(idx + 1) % 2 == 1,
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
        # then apply rotation to the resized high-res image
        deskewed = deskew(resized, denoise_strength=options.denoise_strength, angle_source=img_rgb)

        # Save deskewed image to tmp
        deskew_path = os.path.join(tmp_dir, f"deskew_{page.page_number:04d}.png")
        cv2.imwrite(deskew_path, cv2.cvtColor(deskewed, cv2.COLOR_RGB2BGR))
        page.deskew_file_path = deskew_path

        # Calculate color statistics (Cython with nogil)
        stats = CalculateColorStats(deskewed)
        page.color_stats = stats

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

    # =========================================================================
    # Phase 2: Decide global color adjustment parameters
    # =========================================================================
    report(0, total_pages, "Phase 4.2: Calculating color adjustment parameters...")

    # C# lines 1675-1679: ExcludeOutliers removes top/bottom 20% by MeanR
    filtered_odd_stats = ExcludeOutliers(odd_color_stats)
    filtered_even_stats = ExcludeOutliers(even_color_stats)

    odd_color_param = DecideGlobalColorAdjustment(filtered_odd_stats)
    even_color_param = DecideGlobalColorAdjustment(filtered_even_stats)

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

        # Apply color adjustment (Cython with nogil - modifies in-place)
        color_param = odd_color_param if page.is_odd else even_color_param
        adjusted = np.ascontiguousarray(img_rgb)
        ApplyGlobalColorAdjustment(adjusted, color_param)

        # Save color-adjusted image
        color_adj_path = os.path.join(tmp_dir, f"coloradj_{page.page_number:04d}.png")
        cv2.imwrite(color_adj_path, cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR))
        page.color_adj_file_path = color_adj_path

        # Detect bounding box (Cython - uses OpenCV which releases GIL internally)
        bbox = DetectTextBoundingBox(adjusted)
        page.bounding_box = bbox

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

    # =========================================================================
    # Phase 4: Decide crop regions
    # =========================================================================
    report(0, total_pages, "Phase 4.4: Deciding crop regions...")

    # Cython DecideGroupCropRegion returns (left, top, width, height) tuple
    odd_crop_raw = DecideGroupCropRegion(odd_bboxes)
    even_crop_raw = DecideGroupCropRegion(even_bboxes)


    # Unify crop regions (Cython)
    odd_crop, even_crop = UnifyCropRegions(
        odd_crop_raw, even_crop_raw,
        options.margin_percent,
        INTERNAL_HIGH_RES_WIDTH,
        INTERNAL_HIGH_RES_HEIGHT,
    )


    # Calculate final dimensions
    crop_width = max(odd_crop[2], even_crop[2])
    crop_height = max(odd_crop[3], even_crop[3])

    final_height = FINAL_TARGET_HEIGHT
    final_width = crop_width * final_height // crop_height if crop_height > 0 else crop_width


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

        # Detect vertical writing probability (Cython - C# lines 4693-4722)
        # Use Otsu binarized grayscale image (OpenCV releases GIL internally)
        gray = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v_prob = IsPaperVerticalWriting_GetProbability(binary)

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

    # C# lines 3325-3331: Average probability and threshold at 0.5
    avg_vertical_prob = sum(vertical_probs) / len(vertical_probs) if vertical_probs else 0.0
    is_vertical = avg_vertical_prob >= 0.5

    return {
        'is_vertical': is_vertical,
        'page_offset': page_offset,
        'physical_page_start': physical_page_start,
        'logical_page_start': logical_page_start,
    }
