"""
OCR utilities for page number detection.

Port of C# SuperPdfUtil.cs page number detection (lines 3682-3800).
Uses Tesseract for recognizing page numbers in document margins.
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import cv2

# Try to import pytesseract, make it optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Import Cython functions for text block detection and page number processing
try:
    from .image_processing_cy import (
        OcrGetWordBlocks,
        CalculatePageAlignmentShifts,
        PnOcrProcessForBook,
        PnOcrPageResult,
        PnOcrCandidate,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


def _remove_filled_shapes(binary: np.ndarray) -> np.ndarray:
    """
    C# lines 3823-3940: Remove filled shapes (● ■) from binary image.

    This prevents solid squares/circles from being detected as page numbers.
    Two passes:
    1. Contour analysis to find shapes with fill rate ≥70% and aspect ratio ~1
    2. Distance transform to find thick regions (≥3px)

    Args:
        binary: Binary image (black text on white background, 0/255)

    Returns:
        Binary image with filled shapes removed (painted white)
    """
    result = binary.copy()

    # C# lines 3827-3879: First pass - contour analysis
    # Invert to get white foreground
    inv = cv2.bitwise_not(result)

    # Find contours with hierarchy (for detecting holes)
    contours, hierarchy = cv2.findContours(
        inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is not None and len(contours) > 0:
        hierarchy = hierarchy[0]  # Flatten hierarchy array

        for i, contour in enumerate(contours):
            # Only process outermost contours (parent == -1)
            if hierarchy[i][3] != -1:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 5 or h < 5:
                continue

            # Calculate outer area
            area_outer = cv2.contourArea(contour)

            # Subtract hole areas (children)
            area_hole_sum = 0.0
            child = hierarchy[i][2]  # First child
            while child != -1:
                area_hole_sum += cv2.contourArea(contours[child])
                child = hierarchy[child][0]  # Next sibling

            fill_area = area_outer - area_hole_sum
            area_rect = w * h
            extent = fill_area / (area_rect + 1e-5)
            aspect = w / h

            # C# line 3869: Fill rate ≥70% and aspect ratio 0.75-1.25
            if extent >= 0.70 and 0.75 <= aspect <= 1.25:
                # Paint white using flood fill from center
                seed_x = x + w // 2
                seed_y = y + h // 2
                # Make sure seed is within bounds
                if 0 <= seed_x < result.shape[1] and 0 <= seed_y < result.shape[0]:
                    # Check if seed point is black (0)
                    if result[seed_y, seed_x] == 0:
                        cv2.floodFill(result, None, (seed_x, seed_y), 255)

    # C# lines 3882-3937: Second pass - distance transform for thick regions
    inv2 = cv2.bitwise_not(result)

    # Distance transform
    dist = cv2.distanceTransform(inv2, cv2.DIST_L2, 5)

    # Threshold: pixels with distance ≥ 3 (thick regions)
    _, thick_mask = cv2.threshold(dist, 3.0, 255, cv2.THRESH_BINARY)
    thick_mask = thick_mask.astype(np.uint8)

    # Find contours in thick regions
    contours2, _ = cv2.findContours(
        thick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)

        # C# lines 3913-3915: Filter by size and aspect
        if w < 10 or h < 10:
            continue
        aspect = w / h
        if aspect < 0.80 or aspect > 1.25:
            continue

        area_cnt = cv2.contourArea(contour)
        fill_rate = area_cnt / (w * h + 1e-5)
        if fill_rate < 0.75:
            continue

        # Paint white using flood fill
        seed_x = x + w // 2
        seed_y = y + h // 2
        if 0 <= seed_x < result.shape[1] and 0 <= seed_y < result.shape[0]:
            if result[seed_y, seed_x] == 0:
                cv2.floodFill(result, None, (seed_x, seed_y), 255)

    return result


@dataclass
class PageNumberResult:
    """Result of page number detection.

    shift_x and shift_y are for page alignment based on page number positions.
    In C# (SuperPdfUtil.cs lines 3600-3667), these are calculated by:
    1. Detecting page number bounding boxes
    2. Computing reference points (left edge for left pages, right for right)
    3. Averaging X/Y positions per odd/even group
    4. Setting shift = average - page_position

    Implemented in calculate_page_alignment_shifts_v2 via PnOcrProcessForBook.
    """
    detected_number: Optional[int] = None
    confidence: float = 0.0
    location: str = ""  # 'top', 'bottom', 'left', 'right'
    raw_text: str = ""
    shift_x: int = 0
    shift_y: int = 0
    bbox: Optional[tuple] = None  # (x, y, width, height)


def extract_margin_regions(
    image: np.ndarray,
    margin_percent: float = 10.0,
) -> dict[str, np.ndarray]:
    """
    Extract margin regions from an image for OCR.

    Args:
        image: Input image
        margin_percent: Percentage of image to consider as margin

    Returns:
        Dictionary with 'top', 'bottom', 'left', 'right' margin images
    """
    h, w = image.shape[:2]
    margin_h = int(h * margin_percent / 100)
    margin_w = int(w * margin_percent / 100)

    return {
        'top': image[:margin_h, :],
        'bottom': image[-margin_h:, :],
        'left': image[:, :margin_w],
        'right': image[:, -margin_w:],
    }


def parse_page_number(text: str) -> Optional[int]:
    """
    Try to extract a page number from OCR text.

    Handles various formats:
    - Plain numbers: "42"
    - With decorations: "- 42 -", "[ 42 ]", "( 42 )"
    - Japanese numerals (basic): "四二" → 42

    Args:
        text: Raw OCR text

    Returns:
        Detected page number or None
    """
    text = text.strip()

    # Try plain number
    match = re.search(r'\b(\d+)\b', text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 9999:  # Reasonable page number range
            return num

    # Try Roman numerals (for front matter)
    roman_match = re.search(r'\b([ivxlcdm]+)\b', text.lower())
    if roman_match:
        try:
            roman = roman_match.group(1)
            num = roman_to_int(roman)
            if 1 <= num <= 100:
                return num
        except ValueError:
            pass

    return None


def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    values = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    result = 0
    prev = 0
    for char in reversed(roman.lower()):
        if char not in values:
            raise ValueError(f"Invalid Roman numeral: {roman}")
        curr = values[char]
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


def detect_page_number(
    image: np.ndarray,
    lang: str = 'eng+jpn',
    use_block_detection: bool = True,
) -> PageNumberResult:
    """
    Detect page number from a document image.

    Port of C# OcrDetectPageNumberCandidatesAsync (lines 3682-3800).
    Uses text block detection to find candidate regions, then OCR.

    Args:
        image: Input grayscale or RGB image
        lang: Tesseract language codes
        use_block_detection: Use Cython-based block detection (C# faithful)

    Returns:
        PageNumberResult with detected number, bounding box, and metadata
    """
    if not TESSERACT_AVAILABLE:
        return PageNumberResult()

    h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Use Cython block detection if available (C# faithful)
    if use_block_detection and CYTHON_AVAILABLE:
        return _detect_page_number_with_blocks(gray, lang)

    # Fallback: simple margin-based detection
    return _detect_page_number_simple(gray, lang)


def _detect_page_number_with_blocks(
    gray: np.ndarray,
    lang: str,
) -> PageNumberResult:
    """
    C# faithful page number detection using text block detection.

    Port of OcrDetectPageNumberCandidatesAsync (lines 3682-3800).
    """
    h, w = gray.shape

    # C# lines 3689-3728: Create ignore region (center 66% of page)
    ignore_pct_w = 0.17
    ignore_pct_h = 0.17
    ignore_region = (
        int(w * ignore_pct_w),
        int(h * ignore_pct_h),
        int(w * (1.0 - ignore_pct_w * 2)),
        int(h * (1.0 - ignore_pct_h * 2)),
    )

    # Binarize with Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # C# line 3730: Get word blocks from margins
    word_blocks = OcrGetWordBlocks(binary, ignore_region)

    if not word_blocks:
        # Fallback to simple detection
        return _detect_page_number_simple(gray, lang)

    center_x = w // 2

    # C# lines 3734-3755: Try OCR on each block
    best_result = None
    best_confidence = 0.0

    for bx, by, bw, bh in word_blocks:
        # Skip very large blocks (not page numbers)
        if bw > w * 0.3 or bh > h * 0.15:
            continue

        # Extract block region with padding
        pad = 5
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        block_img = gray[y1:y2, x1:x2]
        if block_img.size == 0:
            continue

        # Binarize block
        _, block_binary = cv2.threshold(block_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR the block
        try:
            config = (
                '--psm 8 '
                '-c tessedit_char_whitelist=0123456789 '
                '-c classify_bln_numeric_mode=1 '
                '-c lstm_choice_mode=2 '
                '-c lstm_choice_iterations=5 '
            )
            text = pytesseract.image_to_string(block_binary, lang=lang, config=config)

            page_num = parse_page_number(text)
            if page_num is not None:
                # Determine location
                if by < h * 0.2:
                    location = 'top'
                elif by > h * 0.8:
                    location = 'bottom'
                elif bx < w * 0.2:
                    location = 'left'
                else:
                    location = 'right'

                # Score: prefer smaller blocks (more likely to be just page number)
                block_area = bw * bh
                confidence = 1.0 / (1.0 + block_area / 10000.0)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = PageNumberResult(
                        detected_number=page_num,
                        confidence=confidence,
                        location=location,
                        raw_text=text.strip(),
                        bbox=(bx, by, bw, bh),
                    )
        except Exception:
            continue

    if best_result is not None:
        return best_result

    # Fallback to simple detection
    return _detect_page_number_simple(gray, lang)


def _detect_page_number_simple(
    gray: np.ndarray,
    lang: str,
) -> PageNumberResult:
    """
    Simple margin-based page number detection (fallback).
    """
    # Extract margin regions
    margins = extract_margin_regions(gray, margin_percent=12)

    # Priority: bottom > top (most books have page numbers at bottom)
    search_order = ['bottom', 'top']

    for location in search_order:
        region = margins[location]

        # Binarize
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR with Tesseract
        try:
            config = (
                '--psm 8 '
                '-c tessedit_char_whitelist=0123456789 '
                '-c classify_bln_numeric_mode=1 '
                '-c lstm_choice_mode=2 '
                '-c lstm_choice_iterations=5 '
            )
            text = pytesseract.image_to_string(binary, lang=lang, config=config)

            page_num = parse_page_number(text)
            if page_num is not None:
                return PageNumberResult(
                    detected_number=page_num,
                    confidence=0.8,
                    location=location,
                    raw_text=text.strip(),
                )
        except Exception:
            continue

    return PageNumberResult()


def detect_page_numbers_batch(
    images: List[np.ndarray],
    lang: str = 'eng+jpn',
) -> List[PageNumberResult]:
    """
    Detect page numbers from multiple images.

    Args:
        images: List of document images
        lang: Tesseract language codes

    Returns:
        List of PageNumberResult for each image
    """
    return [detect_page_number(img, lang) for img in images]


def find_page_number_offset(
    results: List[PageNumberResult],
) -> Optional[int]:
    """
    Determine the offset between physical page index and logical page number.

    For example, if physical page 5 has logical page number 1, offset is 4.

    Args:
        results: List of PageNumberResult from consecutive pages

    Returns:
        Offset value or None if cannot determine
    """
    valid_pairs = []

    for physical_idx, result in enumerate(results):
        if result.detected_number is not None and result.detected_number >= 1:
            offset = physical_idx - (result.detected_number - 1)
            valid_pairs.append(offset)

    if not valid_pairs:
        return None

    # Return most common offset (mode)
    from collections import Counter
    counter = Counter(valid_pairs)
    most_common = counter.most_common(1)[0][0]
    return most_common


def detect_all_page_number_candidates(
    image: np.ndarray,
    physical_page: int,
    lang: str = 'eng',
) -> 'PnOcrPageResult':
    """
    Detect ALL page number candidates from a document image.

    This is the C# faithful implementation that collects all possible
    numbers (not just the best one) for cross-page validation.

    Port of C# OcrDetectPageNumberCandidatesAsync (lines 3632-3750).

    Args:
        image: Input grayscale or RGB image
        physical_page: Physical page number (1-indexed)
        lang: Tesseract language code

    Returns:
        PnOcrPageResult with all number candidates
    """
    if not CYTHON_AVAILABLE:
        # Return empty result if Cython not available
        result = PnOcrPageResult(physical_page)
        return result

    if not TESSERACT_AVAILABLE:
        result = PnOcrPageResult(physical_page)
        return result

    h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]

    if len(image.shape) == 3:
        # C# line 3806: Apply contrast enhancement (1.5x) before grayscale
        # Contrast formula: (pixel - 127.5) * factor + 127.5 (ImageSharp .Contrast())
        contrasted = np.clip((image.astype(np.float32) - 127.5) * 1.5 + 127.5, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(contrasted, cv2.COLOR_RGB2GRAY)
    else:
        # Apply contrast to grayscale (same formula)
        gray = np.clip((image.astype(np.float32) - 127.5) * 1.5 + 127.5, 0, 255).astype(np.uint8)

    # C# lines 3664-3677: Create ignore region (center 66% of page)
    ignore_pct_w = 0.17
    ignore_pct_h = 0.17
    ignore_region = (
        int(w * ignore_pct_w),
        int(h * ignore_pct_h),
        int(w * (1.0 - ignore_pct_w * 2)),
        int(h * (1.0 - ignore_pct_h * 2)),
    )

    # Binarize with Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # C# lines 3823-3940: Remove filled shapes (● ■) from binary image
    binary = _remove_filled_shapes(binary)

    # C# line 3680: Get word blocks from margins
    word_blocks = OcrGetWordBlocks(binary, ignore_region)

    result = PnOcrPageResult(physical_page)

    if not word_blocks:
        return result

    # C# lines 3684-3705: OCR each block and collect ALL number candidates
    for bx, by, bw, bh in word_blocks:
        # Skip very large blocks (not page numbers)
        if bw > w * 0.3 or bh > h * 0.15:
            continue

        # Extract block region with padding
        pad = 5
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        block_img = gray[y1:y2, x1:x2]
        if block_img.size == 0:
            continue

        # Binarize block
        _, block_binary = cv2.threshold(block_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR the block - try to get numbers
        # C# uses these settings (lines 3966-3985):
        #   tessedit_char_whitelist=0123456789
        #   classify_bln_numeric_mode=1  (numeric recognition mode)
        #   lstm_choice_mode=2  (beam search)
        #   lstm_choice_iterations=5
        #   PageSegMode.SingleWord (--psm 8)
        try:
            # Use psm 8 (single word) to match C# PageSegMode.SingleWord
            # Add classify_bln_numeric_mode for better number recognition
            config = (
                '--psm 8 '
                '-c tessedit_char_whitelist=0123456789 '
                '-c classify_bln_numeric_mode=1 '
                '-c lstm_choice_mode=2 '
                '-c lstm_choice_iterations=5 '
            )
            text = pytesseract.image_to_string(block_binary, lang=lang, config=config)
            text = text.strip()

            # Find ALL numbers in the OCR result
            numbers = re.findall(r'\d+', text)
            for num_str in numbers:
                try:
                    num_int = int(num_str)
                    if 1 <= num_int <= 9999:  # Reasonable page number range
                        # Score based on block size (smaller is better for page numbers)
                        block_area = bw * bh
                        possibility = 1.0 / (1.0 + block_area / 10000.0)

                        result.add_candidate(
                            text=num_str,
                            text_int=num_int,
                            bbox=(bx, by, bw, bh),
                            possibility=possibility
                        )
                except ValueError:
                    continue

        except Exception:
            continue

    return result


def calculate_page_alignment_shifts_v2(
    images: List[np.ndarray],
    image_width: int,
    image_height: int,
    progress_callback: Optional[callable] = None,
) -> List[PageNumberResult]:
    """
    Calculate alignment shifts using sophisticated cross-page validation.

    This is the C# faithful implementation (PnOcrProcessForBook) that:
    1. Collects ALL number candidates per page
    2. Cross-validates across pages to find incrementing sequence
    3. Identifies standard bounding box position for page numbers
    4. Calculates shifts based on validated positions

    Port of C# lines 3229-3620 in SuperPdfUtil.cs.

    Args:
        images: List of document images (grayscale or RGB)
        image_width: Width of the page images
        image_height: Height of the page images
        progress_callback: Optional callback(current, total, message) for progress

    Returns:
        List of PageNumberResult with correct shift_x, shift_y, and detected_number
    """
    if not CYTHON_AVAILABLE or not TESSERACT_AVAILABLE:
        # Return zero shifts if dependencies not available
        return [PageNumberResult() for _ in images]

    n = len(images)
    if n == 0:
        return []

    # Phase 1: Collect ALL number candidates from each page
    page_results = []
    for i, image in enumerate(images):
        physical_page = i + 1  # 1-indexed
        page_result = detect_all_page_number_candidates(image, physical_page)
        page_results.append(page_result)

        # Report progress
        if progress_callback:
            progress_callback(i + 1, n, "")

    # Phase 2-4: Cross-page validation and shift calculation
    shift_results = PnOcrProcessForBook(page_results, image_width, image_height)

    # Convert to PageNumberResult list
    results = []
    for i, (shift_x, shift_y, logical_page) in enumerate(shift_results):
        # Get the best candidate for display purposes
        page_result = page_results[i]
        best_text = ""
        best_bbox = None

        if page_result.found_bbox is not None:
            best_bbox = page_result.found_bbox
        elif page_result.candidates:
            # Use first candidate as fallback
            best_bbox = page_result.candidates[0].bbox
            best_text = page_result.candidates[0].text

        results.append(PageNumberResult(
            detected_number=logical_page,
            confidence=1.0 if logical_page is not None else 0.0,
            location='',
            raw_text=best_text,
            shift_x=shift_x,
            shift_y=shift_y,
            bbox=best_bbox,
        ))

    return results


def calculate_page_alignment_shifts(
    results: List[PageNumberResult],
    image_width: int,
) -> List[PageNumberResult]:
    """
    Calculate alignment shifts for all pages based on page number positions.

    DEPRECATED: This function uses the old single-candidate approach.
    Use calculate_page_alignment_shifts_v2 for the C# faithful implementation.

    Port of C# lines 3617-3667 in SuperPdfUtil.cs.

    This function calculates shift_x and shift_y for each page so that
    page numbers align to the same position across all pages.

    Args:
        results: List of PageNumberResult from detect_page_number
        image_width: Width of the page images

    Returns:
        Updated list of PageNumberResult with shift_x and shift_y filled in
    """
    if not CYTHON_AVAILABLE:
        # Return as-is if Cython not available
        return results

    n = len(results)
    if n == 0:
        return results

    # Extract bounding boxes and odd/even flags
    bboxes = [r.bbox for r in results]
    # Physical page numbers are 1-indexed:
    # - index 0 = page 1 (odd), index 1 = page 2 (even), index 2 = page 3 (odd)
    # is_odd_flags[i] = True means physical page (i+1) is odd
    is_odd_flags = [(i % 2) == 0 for i in range(n)]

    # Calculate shifts using Cython function
    shifts = CalculatePageAlignmentShifts(bboxes, is_odd_flags, image_width)

    # Update results with shifts
    updated_results = []
    for i, result in enumerate(results):
        shift_x, shift_y = shifts[i] if i < len(shifts) else (0, 0)
        updated_results.append(PageNumberResult(
            detected_number=result.detected_number,
            confidence=result.confidence,
            location=result.location,
            raw_text=result.raw_text,
            shift_x=shift_x,
            shift_y=shift_y,
            bbox=result.bbox,
        ))

    return updated_results
