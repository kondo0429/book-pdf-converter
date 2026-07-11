"""
Image processing utilities - faithful port of SuperPdfUtil.cs algorithms.

This module implements the exact same algorithms as the C# original:
- Content-aware bounding box detection
- IQR-based outlier removal for crop regions
- Odd/even page grouping
- Color statistics and global color adjustment
- Ghost/bleed suppression with smooth-step whitening
- Natural paper color padding with gradient
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import cv2


# =============================================================================
# Constants (matching C# SuperPdfUtil.cs)
# =============================================================================
INTERNAL_HIGH_RES_WIDTH = 4960
INTERNAL_HIGH_RES_HEIGHT = 7016
FINAL_TARGET_HEIGHT = 3508

# Color adjustment parameters
SAMPLE_STEP = 4  # 1/16 sampling density
SCALE_CLAMP_MIN = 0.8
SCALE_CLAMP_MAX = 4.0
WHITE_CLIP_RANGE = 30
SAT_THRESHOLD = 55
COLOR_DIST_THRESHOLD = 35

# Bounding box detection
BBOX_EDGE_EXCLUSION_PERCENT = 1.0  # Ignore 1% border
BBOX_MIN_AREA_PERCENT = 0.000025  # 0.0025% of total area

# Outlier detection (Tukey fence)
TUKEY_K = 1.5


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class ColorStats:
    """Color statistics for a page (matching C# ColorStats class)."""
    paper_r: float = 255.0
    paper_g: float = 255.0
    paper_b: float = 255.0
    ink_r: float = 0.0
    ink_g: float = 0.0
    ink_b: float = 0.0


@dataclass
class GlobalColorParam:
    """Global color adjustment parameters (matching C# GlobalColorParam class)."""
    scale_r: float = 1.0
    scale_g: float = 1.0
    scale_b: float = 1.0
    offset_r: float = 0.0
    offset_g: float = 0.0
    offset_b: float = 0.0
    ghost_suppress_lum_threshold: int = 200
    white_clip_range: int = 30
    paper_r: int = 255
    paper_g: int = 255
    paper_b: int = 255
    sat_threshold: int = 55
    color_dist_threshold: int = 35


@dataclass
class PageBoundingBox:
    """Bounding box for a single page."""
    page_number: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    is_odd: bool = True


# =============================================================================
# (A) Deskew - Rotation Correction
# =============================================================================
def detect_deskew_angle(image: np.ndarray, max_degree: float = 1.0, threshold_percent: int = 40, denoise_strength: int = 20, border_percent: float = 6.0, vertical_border_percent: float = 10.0, raw_out: Optional[list] = None) -> float:
    """
    Detect the deskew angle of a document image.

    Uses Radon transform for angle detection (ported from ImageMagick's DeskewImage).
    C# lines 2044-2100: GetDeskewRotateDegreeAsync

    Args:
        image: Input image (RGB format) - typically the original extracted PDF page
        max_degree: Maximum angle to correct (degrees), default 1.0 to match C#
        threshold_percent: Deskew threshold percent (default 40 to match C#)
        denoise_strength: Non-local means denoising strength (0 = disabled, default 20)
        border_percent: Percentage of the left/right edges to exclude from
                        detection. Book scans often have dark spine shadows /
                        page edges that binarize as long straight bars and
                        dominate the Radon projection on sparse pages,
                        yielding a wrong angle.
        vertical_border_percent: Percentage of the top/bottom edges to exclude.
                        Running titles / footers are long horizontal text lines
                        that the Radon projection is highly sensitive to; page
                        curvature can leave them level while the body is
                        skewed, so they mask the body skew (detector reports
                        ~0). They sit at the very top/bottom, so excluding 10%
                        keeps the detection on the body text.
        raw_out: Optional list; when given, the raw Radon angle (before the
                 max_degree / near-zero checks and sign negation) is appended.
                 Used for debug output.

    Returns:
        Detected angle in degrees (0.0 if no rotation needed or detection failed)
    """
    from .image_processing_cy import GetDeskewAngle

    # Crop borders so spine shadows / page edges / running titles don't drive
    # the detection. Cropping does not change the skew angle of the remaining
    # content.
    if border_percent > 0 or vertical_border_percent > 0:
        h, w = image.shape[:2]
        mx = int(w * border_percent / 100.0)
        my = int(h * vertical_border_percent / 100.0)
        if w - 2 * mx > 16 and h - 2 * my > 16:
            image = image[my:h - my, mx:w - mx]

    # Apply non-local means denoising before contrast/grayscale (removes noise patterns)
    if denoise_strength > 0:
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, denoise_strength, denoise_strength, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, denoise_strength, 7, 21)
    else:
        denoised = image

    # C# line 2050: First apply Otsu thresholding (for paper page)
    # C# PerformOtsuForPaperPage (lines 3759-3765): Apply contrast enhancement first
    # C# uses ImageSharp's .Contrast(1.5f) then .Grayscale() - all in float internally
    # Contrast formula: (pixel - 127.5) * factor + 127.5
    # Grayscale BT.709: Y = 0.2126×R + 0.7152×G + 0.0722×B
    def apply_contrast_and_grayscale_bt709(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Apply contrast then BT.709 grayscale in float, convert to uint8 only at end.
        This matches ImageSharp's internal float pipeline to avoid precision loss.
        """
        img_f = img.astype(np.float32)
        # Contrast around midpoint 127.5 (exact ImageSharp formula)
        contrasted = (img_f - 127.5) * factor + 127.5
        # Grayscale BT.709 (no intermediate uint8 conversion)
        gray = 0.2126 * contrasted[:, :, 0] + 0.7152 * contrasted[:, :, 1] + 0.0722 * contrasted[:, :, 2]
        return np.clip(gray, 0, 255).astype(np.uint8)

    def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
        """Apply contrast adjustment around midpoint (127.5), matching ImageSharp."""
        return np.clip((img.astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)

    if len(denoised.shape) == 3:
        gray = apply_contrast_and_grayscale_bt709(denoised, 1.5)
    else:
        gray = apply_contrast(denoised, 1.5)

    # Erase long straight line segments (chart axes and trend lines, table
    # rules, box borders) before the Radon projection: such a line produces a
    # sharp peak at its OWN angle and can hijack the page angle (e.g. a graph
    # trend arrow at 6 deg rotating the whole page). Text never forms long
    # straight segments, so this only removes non-text structures - and only
    # from the detection input, not from the page itself.
    gh, gw = gray.shape[:2]
    edges_img = cv2.Canny(gray, 50, 150)
    min_len = int(min(gh, gw) * 0.15)
    lines = cv2.HoughLinesP(edges_img, 1, np.pi / 360, threshold=120,
                            minLineLength=min_len, maxLineGap=8)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            cv2.line(gray, (int(x1), int(y1)), (int(x2), int(y2)), 255, 15)

    # Use Cython port of ImageMagick's deskew angle detection (Radon transform)
    # The algorithm does its own binarization using threshold_percent
    angle = GetDeskewAngle(gray, threshold_percent / 100.0)
    if raw_out is not None:
        raw_out.append(angle)

    # C# lines 362-365: If angle exceeds max_degree, return 0 (no rotation)
    if abs(angle) > max_degree:
        return 0.0

    # C# line 2059: Check if nearly zero
    if abs(angle) < 0.001:
        return 0.0

    # C# line 2065: Negate the angle (ImageMagick returns opposite sign)
    # The Cython GetDeskewAngle returns the same value as ImageMagick's output,
    # so we negate it here to match C# behavior.
    return -angle


def apply_deskew_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Apply rotation to deskew an image.

    C# lines 2071-2084: Rotate using OpenCV with white background.

    Args:
        image: Input image (RGB format)
        angle: Rotation angle in degrees

    Returns:
        Rotated image
    """
    if abs(angle) < 0.001:
        return image

    h, w = image.shape[:2]

    # C# lines 2071-2084: Rotate using OpenCV with white background
    # C# rotates a 4-channel RGBA image, so we do the same for matching results
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Convert RGB to RGBA before rotation (add alpha channel)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        border_value = (255, 255, 255, 255)  # 4-channel RGBA white
    else:
        image_rgba = image
        border_value = (255, 255, 255, 255) if len(image.shape) == 3 else 255

    rotated_rgba = cv2.warpAffine(
        image_rgba, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    # Convert back to RGB after rotation
    if len(image.shape) == 3 and image.shape[2] == 3:
        rotated = cv2.cvtColor(rotated_rgba, cv2.COLOR_RGBA2RGB)
    else:
        rotated = rotated_rgba

    return rotated


def deskew(image: np.ndarray, max_degree: float = 1.0, threshold_percent: int = 40, denoise_strength: int = 20, angle_source: np.ndarray = None) -> np.ndarray:
    """
    Deskew (straighten) a rotated document image.

    Uses Radon transform for angle detection (Cython port of ImageMagick's algorithm).
    C# lines 2044-2100: DeskewImageWithOpenCvAsync + GetDeskewRotateDegreeAsync

    Args:
        image: Input image to rotate (RGB format)
        max_degree: Maximum angle to correct (degrees), default 1.0 to match C#
        threshold_percent: Deskew threshold percent (default 40 to match C#)
        denoise_strength: Non-local means denoising strength (0 = disabled, default 20)
        angle_source: Optional separate image to use for angle detection (e.g., original low-res).
                      If None, uses `image` for both detection and rotation.

    Returns:
        Deskewed image
    """
    # Detect angle from angle_source if provided, otherwise from image
    detection_image = angle_source if angle_source is not None else image
    angle = detect_deskew_angle(detection_image, max_degree, threshold_percent, denoise_strength)

    # Apply rotation to the main image
    return apply_deskew_rotation(image, angle)


# =============================================================================
# (A-2) Show-through / bleed-through removal (local flat-field + whitening)
# =============================================================================
def remove_show_through(
    image: np.ndarray,
    bg_ksize: int = 151,
    black_point: int = 115,
    white_point: int = 205,
    gamma: float = 1.0,
    mask_threshold: int = 200,
    edge_pad: int = 5,
    soft_white_point: int = 235,
) -> np.ndarray:
    """
    Remove show-through (裏映り) text and non-uniform background color.

    Unlike the global-linear ApplyGlobalColorAdjustment, this estimates the paper
    background *locally* and flattens it, so uneven illumination and faint
    reverse-side text are eliminated while foreground ink is preserved. The output
    is grayscale replicated to 3 channels (RGB), suitable for text-heavy pages.

    Steps:
      1. Estimate paper background via morphological closing (removes dark text,
         keeps illumination/paper) followed by a Gaussian blur.
      2. Flat-field: divide the image by the background so paper -> ~255 everywhere.
      3. Hard contrast stretch between black_point and white_point. This alone
         gives a clean white background but clips the anti-aliased fringes of
         glyphs, visually thinning the strokes - so it is used only to DECIDE
         what matters, not as the final rendering.
      4. Composite: pixels the hard stretch keeps dark form an importance mask;
         the mask is dilated by edge_pad so glyph fringes are included, and
         inside it the gently stretched flat-field tones (black_point ..
         soft_white_point, no harsh white clipping) are used. Everything
         outside the mask becomes pure white. This keeps the background
         perfectly white while preserving the smooth anti-aliased edges of
         text and image detail.

    Args:
        image: Input image (RGB format)
        bg_ksize: Background estimation kernel size (odd). Must be larger than the
                  thickest stroke / character spacing, or text gets eaten.
        black_point: Values <= this (after flat-field) become pure ink (0).
        white_point: Values >= this (after flat-field) count as background;
                     lower it to remove show-through more aggressively.
        gamma: Optional gamma applied to the stretched tones (1.0 = linear).
        mask_threshold: Hard-stretch values below this mark important content.
        edge_pad: Dilation radius (px) of the importance mask, so anti-aliased
                  glyph fringes survive around every stroke.
        soft_white_point: White point of the gentle foreground stretch (higher
                          than white_point = smoother edges inside the mask).

    Returns:
        Show-through-removed image (RGB, 3-channel grayscale)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 1) Estimate paper background (dark text closed out, illumination retained)
    if bg_ksize % 2 == 0:
        bg_ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_ksize, bg_ksize))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), bg_ksize / 6.0)

    # 2) Flat-field division: paper -> ~255 uniformly across the page
    norm = gray.astype(np.float32) / (bg.astype(np.float32) + 1e-3) * 255.0

    # 3) Hard contrast stretch: ink -> 0, paper + show-through -> 255.
    #    Used only to decide what is important content (it clips glyph
    #    fringes, so rendering it directly thins the strokes).
    denom = max(white_point - black_point, 1)
    hard = np.clip((norm - black_point) / denom, 0.0, 1.0)
    if gamma != 1.0:
        hard = hard ** gamma
    hard8 = (hard * 255.0).astype(np.uint8)

    # 4) Importance mask (content the hard stretch keeps visibly dark),
    #    dilated a little so the anti-aliased fringes of glyphs are covered
    mask = (hard8 < mask_threshold).astype(np.uint8)
    if edge_pad > 0:
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * edge_pad + 1, 2 * edge_pad + 1))
        )

    # Gentle foreground tones: same black point, but a much higher white
    # point, so glyph edge gradients survive instead of being clipped away
    soft_denom = max(soft_white_point - black_point, 1)
    soft = np.clip((norm - black_point) / soft_denom, 0.0, 1.0)
    if gamma != 1.0:
        soft = soft ** gamma
    soft8 = (soft * 255.0).astype(np.uint8)

    # Composite: white background, smooth tones inside the (padded) mask
    out = np.full_like(hard8, 255)
    m = mask > 0
    out[m] = soft8[m]

    # Return as 3-channel RGB so downstream (bbox, crop, PDF) stays uniform
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


# =============================================================================
# (A-2b) Page-edge junk detection (photography stand / page-edge shadows)
# =============================================================================
def detect_page_edge_junk(
    image: np.ndarray,
    border: int = 16,
    min_area: int = 500,
    run_frac: float = 0.08,
    max_bar_thickness: int = 150,
    dark_ratio: float = 0.8,
    vbar_dark_ratio: float = 0.93,
    vbar_ink_dark: int = 90,
    vbar_ink_max_frac: float = 0.06,
    debug_out: Optional[dict] = None,
) -> np.ndarray:
    """
    Detect scan junk (photography stand slabs, page-edge / fold / spine
    shadows) on the PRE-color-adjustment image, where they are still solid
    dark shapes.

    The show-through flat-field neutralizes the interior of large dark slabs
    into mottled texture that mimics text density, so junk must be located
    BEFORE that step. Three rules, all physically impossible for print:
    (1) dark regions touching the image border - the padded internal image
        never has print at its border (photography stand / scan bed)
    (2) thin continuous dark bars (>= run_frac of the image dimension long,
        <= max_bar_thickness thick) - text always breaks between characters,
        so such runs can't be text (page-edge / fold shadow lines).
    (3) thin continuous VERTICAL bars at a fainter threshold (vbar_dark_ratio)
        - smooth low-contrast spine/gutter shadows and reverse-side
        show-through columns. Real text never forms such a bar (characters
        leave gaps), and as an extra guard any candidate that actually
        contains dark ink (>= vbar_ink_max_frac of pixels darker than
        vbar_ink_dark) is skipped, so a dense text column near the edge is
        never masked.
    The mask is only used to keep these structures from anchoring the text
    extent (they are excluded from the text detection); it does not paint
    anything, so nothing that is real text can be erased.
    Photos are safe: they don't touch the border and are thick in both
    directions.

    Args:
        image: Deskewed page image BEFORE color adjustment (RGB or grayscale)
        border: Border contact distance in px for rule (1)
        min_area: Minimum component area for rule (1)
        run_frac: Minimum continuous run length as a fraction of height/width
        max_bar_thickness: Maximum thickness for the bar rules
        dark_ratio: Pixels darker than paper * dark_ratio count as dark for
                    rules (1) and (2)
        vbar_dark_ratio: Fainter threshold for rule (3) so smooth low-contrast
                         spine/gutter shadows are caught
        vbar_ink_dark: Pixels darker than this count as real ink for rule (3)
        vbar_ink_max_frac: Skip a rule-(3) bar whose real-ink fraction exceeds
                           this (it is a text column, not a shadow)

    Returns:
        uint8 bit-flag mask in the same geometry as `image`: nonzero = junk;
        bit 1 = bar/strip junk (rules 2-3), bit 2 = border-connected junk
        (rule 1; includes neighboring-page print bleeding past the page edge,
        which must never be rescued as text)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape

    # Paper estimate from the page center (robust to edge junk; on photo pages
    # it under-estimates, which only makes the junk rules more conservative)
    paper = float(np.median(gray[int(h * 0.25):int(h * 0.75),
                                 int(w * 0.35):int(w * 0.85)]))
    dark_thr = max(60.0, paper * dark_ratio)
    dark = (gray < dark_thr).astype(np.uint8)

    junk = np.zeros((h, w), dtype=np.uint8)
    t0 = time.perf_counter()

    # (1) border-connected dark regions. Marked with bit 2 (value 2) so
    #     callers can tell them apart: print from the NEIGHBORING page visible
    #     beyond this page's edge merges with the dark gutter into a
    #     border-connected blob, and must never be rescued as text, while
    #     this page's own glyphs never touch the border.
    #     (Selection is vectorized via a per-label lookup table - a Python
    #     loop with a full-size `lab == i` per component is far too slow.)
    num, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    sel = ((st[:, cv2.CC_STAT_AREA] >= min_area)
           & ((st[:, cv2.CC_STAT_LEFT] < border)
              | (st[:, cv2.CC_STAT_TOP] < border)
              | (st[:, cv2.CC_STAT_LEFT] + st[:, cv2.CC_STAT_WIDTH] > w - border)
              | (st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT] > h - border)))
    sel[0] = False
    junk[sel[lab]] |= 2
    t1 = time.perf_counter()

    # (2) thin continuous bars (vertical then horizontal)
    run_v = max(int(h * run_frac), 1)
    bars = cv2.dilate(cv2.erode(dark, np.ones((run_v, 1), np.uint8)),
                      np.ones((run_v, 1), np.uint8))
    num_v, lab_v, st_v, _ = cv2.connectedComponentsWithStats(bars, 8)
    sel = st_v[:, cv2.CC_STAT_WIDTH] <= max_bar_thickness
    sel[0] = False
    junk[sel[lab_v]] |= 1

    run_h = max(int(w * run_frac), 1)
    bars = cv2.dilate(cv2.erode(dark, np.ones((1, run_h), np.uint8)),
                      np.ones((1, run_h), np.uint8))
    num_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(bars, 8)
    sel = st_h[:, cv2.CC_STAT_HEIGHT] <= max_bar_thickness
    sel[0] = False
    junk[sel[lab_h]] |= 1
    t2 = time.perf_counter()

    # (3) thin continuous vertical bars at the fainter vbar threshold: smooth
    #     spine/gutter shadows and show-through columns. Skip any candidate
    #     that carries real ink (a dense text column), so text is never masked.
    dark_v = (gray < max(60.0, paper * vbar_dark_ratio)).astype(np.uint8)
    bars = cv2.dilate(cv2.erode(dark_v, np.ones((run_v, 1), np.uint8)),
                      np.ones((run_v, 1), np.uint8))
    num_v, lab_v, st_v, _ = cv2.connectedComponentsWithStats(bars, 8)
    flat_v = lab_v.ravel()
    sizes = np.bincount(flat_v, minlength=num_v)
    ink_cnt = np.bincount(flat_v, weights=(gray < vbar_ink_dark).ravel(),
                          minlength=num_v)
    sel = ((st_v[:, cv2.CC_STAT_WIDTH] <= max_bar_thickness)
           & ((ink_cnt / np.maximum(sizes, 1)) <= vbar_ink_max_frac))
    sel[0] = False
    junk[sel[lab_v]] |= 1

    if debug_out is not None:
        t3 = time.perf_counter()
        debug_out['t_junk_r1'] = t1 - t0
        debug_out['t_junk_r2'] = t2 - t1
        debug_out['t_junk_r3'] = t3 - t2
        debug_out['junk_r1_comps'] = int(num - 1)
        debug_out['junk_r3_comps'] = int(num_v - 1)

    return junk


# =============================================================================
# (A-3) Margin background whitening (text-free bands touching the page edges)
# =============================================================================
def remove_margin_background(
    image: np.ndarray,
    stroke_threshold: int = 22,
    ink_threshold: int = 145,
    dens_win: int = 31,
    dens_threshold: float = 0.015,
    min_area_frac: float = 0.00005,
    margin_pad: int = 40,
    edge_exclude_frac: float = 0.10,
    paper: int = 255,
    junk_mask: Optional[np.ndarray] = None,
    debug_out: Optional[dict] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Whiten the four outer margin bands that contain no text.

    A band qualifies for deletion only if it touches a page edge and runs to
    the opposite edge without any text: everything left of the leftmost text,
    right of the rightmost text, above the topmost text, and below the
    bottommost text. Anything sharing rows/columns with text (rules, figures,
    dark junk between text columns) is never touched, so text that the
    detector only partially finds can no longer be erased the way zone-based
    clearing could.

    Text is detected as dense sharp dark strokes (Laplacian response on dark
    pixels). Spine shadows / page-edge streaks can have sharp outlines that
    mimic strokes, so components lying entirely within the outer
    edge_exclude_frac of the width are treated as edge junk, not text
    (real text never sits at the extreme edge of the padded internal image).

    Args:
        image: Input image (RGB or grayscale)
        stroke_threshold: Laplacian magnitude above which a dark pixel is a stroke.
        ink_threshold: Pixels darker than this can count as stroke pixels.
        dens_threshold: Min stroke density for a pixel to count as text
                        (kept low so short/faint headers are still protected).
        min_area_frac: Text components smaller than this fraction of the image
                       are ignored (drops speckle, keeps small headers).
        margin_pad: Extra pixels kept around the detected text extent.
        edge_exclude_frac: Components fully inside the outer left/right strip of
                           this width fraction never count as text.
        paper: Replacement value (255 = white).
        junk_mask: Optional mask from detect_page_edge_junk() (computed on the
                   PRE-adjustment image, same geometry): marked areas never
                   count as text, so the margin bands extend over them.

    Returns:
        Tuple of:
        - Image with the text-free outer margin bands painted to paper.
        - Text extent (left, top, right, bottom) including margin_pad — the
          inner boundaries of the painted bands — or None when no text was
          found (image returned unchanged). The band widths derived from this
          extent tell the caller how much page-edge area was cleared, so
          equivalent margins can be re-granted around the final crop.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    h, w = gray.shape
    total = float(h * w)
    _t0 = time.perf_counter()

    # Text mask: dense sharp dark strokes (smooth shadows have low response)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    stroke = ((lap > stroke_threshold) & (gray < ink_threshold)).astype(np.float32)
    dens = cv2.boxFilter(stroke, -1, (dens_win, dens_win), normalize=True)
    text = (dens > dens_threshold).astype(np.uint8)

    # Drop tiny speckle components and edge-strip junk; keep anything that
    # could be a header
    num, labels, stats, _ = cv2.connectedComponentsWithStats(text, 8)
    strip = edge_exclude_frac * w
    kept_ids = []
    strip_ids = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < min_area_frac * total:
            continue
        # Fully inside the outer left/right strip -> edge junk, not text -
        # but such a component may still carry real glyphs (a running head
        # hugging the fore-edge), so it stays eligible for the ink rescue
        if x + bw <= strip or x >= w - strip:
            strip_ids.append(i)
            continue
        kept_ids.append(i)

    _t1 = time.perf_counter()

    # Junk detected on the PRE-adjustment image (see detect_page_edge_junk):
    # photography-stand slabs and page-edge shadow lines must not count as
    # text, or the margin bands stop at them and the junk survives. It has to
    # be detected before the show-through flat-field (which turns solid slabs
    # into mottled texture that mimics text density), so the caller passes the
    # mask in.
    keep = np.zeros((h, w), dtype=bool)
    if junk_mask is not None and junk_mask.any():
        # Fatten past the box-filter halo so the stroke-density ring around a
        # junk structure is removed from the text mask as well
        fat = cv2.dilate(
            junk_mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dens_win + 1, 2 * dens_win + 1))
        ) > 0
        # Component-aware subtraction: a text component mostly swallowed by
        # the junk halo is dropped, but the glyph-sized real-ink parts inside
        # it are rescued individually - a page number or a running head
        # sitting on (or merged with) a faint show-through column / page-edge
        # shadow must survive, while junk halos (no ink) and shadow cores
        # (tall continuous ink) are still dropped. Rescue is per ink part, so
        # a tall shadow line merged into the same density component cannot
        # veto the glyphs next to it. Ink lying on border-connected junk
        # (bit 2) is never rescued: that is print from the NEIGHBORING page
        # visible beyond this page's edge, not this page's text.
        ink = gray < 100
        border_junk = cv2.dilate(
            ((junk_mask & 2) > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        glyph_max_h = int(h * 0.05)
        glyph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dens_win, dens_win))

        def rescue_glyphs(comp: np.ndarray):
            """Keep the glyph-sized, non-border ink parts of a junk component."""
            comp_ink = (comp & ink).astype(np.uint8)
            if int(comp_ink.sum()) < 200:
                return
            num_k, lab_k, st_k, _ = cv2.connectedComponentsWithStats(comp_ink, 8)
            # vectorized per-part selection (see _glyph_ink_mask, incl. the
            # fill-ratio gate against solid shadow lumps)
            bbox_k = np.maximum(st_k[:, cv2.CC_STAT_WIDTH].astype(np.int64)
                                * st_k[:, cv2.CC_STAT_HEIGHT], 1)
            ok = ((st_k[:, cv2.CC_STAT_HEIGHT] <= glyph_max_h)
                  & (st_k[:, cv2.CC_STAT_AREA] >= 50)
                  & (st_k[:, cv2.CC_STAT_AREA] / bbox_k <= 0.65))
            ok[0] = False
            if ok.any():
                flat_k = lab_k.ravel()
                sizes_k = np.bincount(flat_k, minlength=num_k)
                bj_k = np.bincount(flat_k, weights=border_junk.ravel(),
                                   minlength=num_k)
                ok &= (bj_k / np.maximum(sizes_k, 1)) <= 0.3
            if ok.any():
                sel = ok[lab_k].astype(np.uint8)
                # restore the glyphs with their density halo
                nonlocal keep
                keep |= cv2.dilate(sel, glyph_kernel).astype(bool) & comp

        # Per-component junk-halo coverage, computed in one pass over the
        # label image (per-component boolean masks are far too slow)
        flat_l = labels.ravel()
        sizes_l = np.bincount(flat_l, minlength=num)
        cover_l = np.bincount(flat_l, weights=fat.ravel(), minlength=num)
        cover_frac = cover_l / np.maximum(sizes_l, 1)

        clear = np.zeros(num, dtype=bool)
        swallowed = []
        for i in kept_ids:
            if cover_frac[i] <= 0.5:
                clear[i] = True  # mostly clear: keep, trimming junk-covered part
            else:
                swallowed.append(i)
        keep |= clear[labels] & ~fat
        for i in swallowed:
            rescue_glyphs(labels == i)
        # Strip components may hold a fore-edge running head - but only on a
        # CLEAN side. A side whose strip zone carries substantial
        # border-connected junk is the gutter/spine side, where glyph-shaped
        # ink is print from the neighboring page, never this page's text.
        strip_px = max(int(strip), 1)
        bj = (junk_mask & 2) > 0
        side_bj = {
            'L': float(bj[:, :strip_px].mean()),
            'R': float(bj[:, w - strip_px:].mean()),
        }
        for i in strip_ids:
            side = 'L' if stats[i, cv2.CC_STAT_LEFT] + \
                stats[i, cv2.CC_STAT_WIDTH] <= strip else 'R'
            if side_bj[side] > 0.02:
                continue
            rescue_glyphs(labels == i)
    else:
        sel = np.zeros(num, dtype=bool)
        sel[kept_ids] = True
        keep = sel[labels]

    # Validate the left/right extent anchors. Junk fragments that every
    # detector missed (partial junk coverage, odd shapes) can still anchor
    # the extent at the page edge, blocking the margin bands. Real content
    # anchoring an extent always shows glyph ink (text, page numbers,
    # running heads - checked with the solid-lump-rejecting fill gate) or is
    # a photo-sized 2-D region. Trim keep in the outer zones down to the
    # first glyph-bearing column, sparing photo-sized components.
    # min_area=300: an anchor must be backed by substantial glyph pieces -
    # page-edge line fragments and halftone/crease speckle are glyph-SHAPED
    # but small (<300 px each), while real characters/digits are larger
    anchor_glyphs = _glyph_ink_mask(gray, junk_mask, min_area=300)
    # Strong-glyph columns: enough glyph ink within a one-column-wide window
    # that the anchor is a real text column / running head, not stray marks
    col_px = anchor_glyphs.sum(axis=0).astype(np.float32)
    win_px = cv2.blur(col_px.reshape(1, -1), (121, 1)).ravel() * 121
    gcols = np.where(win_px >= 600)[0]
    if len(gcols) and keep.any():
        trim_cap = int(w * 0.15)
        big = ((stats[:, cv2.CC_STAT_WIDTH] >= int(w * 0.08))
               & (stats[:, cv2.CC_STAT_HEIGHT] >= int(h * 0.08))
               & (stats[:, cv2.CC_STAT_AREA] >= int(0.002 * total)))
        big[0] = False
        protected = big[labels]
        kc = np.where(keep.any(axis=0))[0]
        gl_first, gl_last = int(gcols[0]), int(gcols[-1])
        if kc[0] < gl_first:
            end = min(gl_first, trim_cap)
            if end > 0:
                keep[:, :end] &= protected[:, :end]
        if kc[-1] > gl_last:
            start = max(gl_last + 1, w - trim_cap)
            if start < w:
                keep[:, start:] &= protected[:, start:]

    _t2 = time.perf_counter()

    if not keep.any():
        # No text found (blank or photo-only page misdetection) - do nothing
        return image, None

    rows = np.where(keep.any(axis=1))[0]
    cols = np.where(keep.any(axis=0))[0]
    top = max(0, int(rows[0]) - margin_pad)
    bottom = min(h, int(rows[-1]) + 1 + margin_pad)
    left = max(0, int(cols[0]) - margin_pad)
    right = min(w, int(cols[-1]) + 1 + margin_pad)

    out = image.copy()
    out[:top] = paper
    out[bottom:] = paper
    out[:, :left] = paper
    out[:, right:] = paper

    # Whiten residual spine/gutter-shadow smudges. The flat-field turns a thin
    # smooth binding shadow into wider lens-shaped smudges that carry dark
    # cores; those cores cross the edge-exclude boundary, get kept as text, and
    # re-anchor the margin band so the band stops short of them. They are still
    # SMOOTH (low Laplacian) unlike real ink strokes, so they are removed here
    # by detecting smooth dark vertical smudges in the outer side margins and
    # painting them out. Confined to the central height band (page numbers /
    # running heads in the top/bottom corners are handled by the bands and left
    # alone) and to the outer side zones (body text is never touched); the
    # smoothness gate keeps real edge text, which is sharp, safe.
    _t3 = time.perf_counter()
    out = _whiten_shadow_smudges(out, paper_value=paper, junk_mask=junk_mask,
                                 debug_out=debug_out)
    _t4 = time.perf_counter()

    # Whiten fore-edge page-stack shadows: a dark or faint band hugging the
    # page edge - the exposed stack of page edges. It is neither smooth (so
    # the smudge pass misses it) nor free of ink (so the junk pass skips it),
    # and it re-anchors the text extent, so the margin bands stop right at it.
    # Search therefore starts at the freshly painted band boundaries (the text
    # extent edges), walking inward through the depressed brightness band.
    out = _whiten_edge_dark_band(out, paper_value=paper,
                                 anchor_l=left, anchor_r=right,
                                 junk_mask=junk_mask,
                                 debug_out=debug_out)

    # Final faint-margin cleanup. Everything above targets DARK structures;
    # very faint low-contrast marks in the side margins (fold/edge shadow
    # lines at gray ~200-235, show-through speckle, tick/dash marks) slip
    # under every dark threshold yet stay visible, and their broken segments
    # can even mimic isolated glyphs. In the CENTRAL height band of the outer
    # side zones, whiten everything that is not (a) part of a DENSE glyph
    # cluster - real body text is dense, stray marks are sparse - or (b) a
    # photo. The top/bottom head_foot bands are left untouched, so running
    # heads (top) and page numbers (bottom), which are legitimately sparse,
    # are never scrubbed. Body text reaching into the side zone stays because
    # it is dense; the sparse faint lines and dashes go.
    og = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) if out.ndim == 3 else out
    # Only FAINT pixels are eligible for scrubbing. Real text is dark, and
    # dark pixels are never touched here, so no character can be erased even
    # when a body-text column runs right up to the page edge; the faint marks
    # (low-contrast fold/edge shadow lines, show-through speckle, dashes) are
    # what gets removed.
    faint = (og >= 160) & (og < 250)
    # Protect the faint anti-aliasing halos of REAL text: where dark ink is
    # locally dense (a text column or body block, however close to the edge),
    # a dilated region is spared. Isolated marks have only thin sparse dark
    # cores, so they never reach this density and their faint bodies go.
    dark = (og < 130).astype(np.uint8)
    ddens = cv2.boxFilter(dark, -1, (121, 121), normalize=False)
    dense_dark = cv2.dilate((ddens >= 1500).astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0
    # Photo guard: a large 2-D dark region (illustration bleeding into the
    # margin) must not be scrubbed.
    darkish = (og < 200).astype(np.uint8)
    nP, labP, stP, _ = cv2.connectedComponentsWithStats(darkish, 8)
    photo = np.zeros(nP, dtype=bool)
    photo[1:] = ((stP[1:, cv2.CC_STAT_WIDTH] >= int(w * 0.15))
                 & (stP[1:, cv2.CC_STAT_HEIGHT] >= int(h * 0.15))
                 & (stP[1:, cv2.CC_STAT_AREA] >= int(0.02 * total)))
    preserve = dense_dark | photo[labP]

    zone = int(w * 0.15)
    scrub = np.zeros((h, w), dtype=bool)
    scrub[:, :zone] = True
    scrub[:, w - zone:] = True
    scrub &= faint & ~preserve
    out[scrub] = paper

    if debug_out is not None:
        _t5 = time.perf_counter()
        debug_out['t_textmask'] = _t1 - _t0
        debug_out['t_subtract'] = _t2 - _t1
        debug_out['t_bands'] = _t3 - _t2
        debug_out['t_smudge'] = _t4 - _t3
        debug_out['t_edgeband'] = _t5 - _t4
        debug_out['text_comps'] = int(num - 1)
        debug_out['text_kept'] = len(kept_ids)
        debug_out['text_strip'] = len(strip_ids)

    return out, (left, top, right, bottom)


def _glyph_ink_mask(
    gray: np.ndarray,
    junk_mask: Optional[np.ndarray],
    ink_keep: int = 100,
    glyph_max_frac: float = 0.05,
    min_area: int = 50,
    border_overlap_max: float = 0.3,
) -> np.ndarray:
    """Mask of glyph-sized real-ink parts that are NOT on border-connected junk.

    These are the ink pieces the side-margin passes must treat as text (page
    numbers, running heads): small enough to be glyphs, and not lying on
    border-connected junk (bit 2 of junk_mask) - ink there is either print
    from the neighboring page or dark specks inside a page-stack shadow band,
    neither of which is this page's text.
    """
    h, w = gray.shape[:2]
    ink = (gray < ink_keep).astype(np.uint8)
    out = np.zeros((h, w), dtype=bool)
    if not ink.any():
        return out
    bj = None
    if junk_mask is not None:
        # Generous dilation: border-connected junk (page-stack bands, gutter
        # shadows) is patchy at the dark threshold, and ink speckle inside it
        # falls in the gaps - it must still count as junk, not as glyphs.
        # Only bit 2 (border-connected) disqualifies: text legitimately sits
        # on top of faint show-through columns (bit 1).
        bj = cv2.dilate(((junk_mask & 2) > 0).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    glyph_max_h = int(h * glyph_max_frac)
    num, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    # Vectorized per-component selection (a Python loop with a full-size
    # `lab == i` per glyph is O(components * pixels) and dominated Phase 4.3)
    # Fill-ratio gate: glyphs are strokes with paper inside (fill well below
    # 1), while shadow blobs / edge fragments are solid lumps - dark AND
    # glyph-sized, but not text.
    bbox_area = np.maximum(st[:, cv2.CC_STAT_WIDTH].astype(np.int64)
                           * st[:, cv2.CC_STAT_HEIGHT], 1)
    fill = st[:, cv2.CC_STAT_AREA] / bbox_area
    ok = ((st[:, cv2.CC_STAT_HEIGHT] <= glyph_max_h)
          & (st[:, cv2.CC_STAT_AREA] >= min_area)
          & (fill <= 0.65))
    ok[0] = False  # background label
    if bj is not None and ok.any():
        flat = lab.ravel()
        sizes = np.bincount(flat, minlength=num)
        bj_cnt = np.bincount(flat, weights=bj.ravel(), minlength=num)
        ok &= (bj_cnt / np.maximum(sizes, 1)) <= border_overlap_max
    return ok[lab]


def _whiten_edge_dark_band(
    image: np.ndarray,
    paper_value=255,
    anchor_l: int = 0,
    anchor_r: Optional[int] = None,
    recover_ratio: float = 0.93,
    white_recover: int = 252,
    smooth_frac: float = 0.015,
    cap_frac: float = 0.18,
    anchor_frac: float = 0.04,
    gap_frac: float = 0.012,
    head_foot_frac: float = 0.10,
    ink_keep: int = 100,
    edge_zone_frac: float = 0.18,
    junk_mask: Optional[np.ndarray] = None,
    debug_out: Optional[dict] = None,
) -> np.ndarray:
    """Paint out a dark band anchored to the outer margin boundary.

    Targets fore-edge page-stack shadows: a band (textured, smooth, or faint)
    that hugs the page edge. Such a band re-anchors the text extent, so the
    margin bands whiten only up to it and the band itself survives - therefore
    the search starts at the band boundary (anchor_l / anchor_r, the text
    extent edges the caller just painted up to), not at the image border.
    Starting within anchor_frac of that boundary, the contiguous run of
    columns below the recovery threshold (small bright gaps up to gap_frac
    are bridged) is painted out, capped at cap_frac.

    Content is protected three ways. (1) The strip may only cover columns in
    the outer edge_zone_frac of the image width: stack shadows live at the
    page edge, while charts / figures / text columns live in the content
    area, so the walk can never reach them even when their columns depress
    the brightness. (2) The walk stops at the first column that contains
    glyph-sized ink (text labels, running heads). (3) The strip is only
    painted when it contains a column with a CONTINUOUS non-white vertical
    run that text can never produce (characters leave gaps), so a strip that
    is actually the text block is rejected; and within a painted strip,
    glyph-sized ink components are preserved while tall shadow cores are
    removed.

    On flat-fielded pages the background is exactly white, so any column mean
    below white_recover marks the band (catches even very faint stack
    shadows). On non-whitened (color) pages a conservative paper*recover_ratio
    is used instead.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape
    if anchor_r is None:
        anchor_r = w
    paper = float(np.median(gray[int(h * 0.25):int(h * 0.75),
                                 int(w * 0.35):int(w * 0.65)]))
    hf = int(h * head_foot_frac)
    central = gray[hf:h - hf, :]
    col = central.astype(np.float32).mean(axis=0)
    k = (int(w * smooth_frac) // 2) * 2 + 1
    cs = cv2.blur(col.reshape(1, -1), (k, 1)).ravel()
    cap = int(w * cap_frac)
    anchor = int(w * anchor_frac)
    gap = max(int(w * gap_frac), 1)
    edge_zone = int(w * edge_zone_frac)
    if paper >= 250:
        rec = float(white_recover)
    else:
        rec = paper * recover_ratio

    # Columns containing glyph-sized ink (text labels / running heads): the
    # strip walk must never cross them. Tall ink (shadow cores) does not
    # count - the walk can pass over a solid stack-shadow core - and neither
    # does ink on border-connected junk (dark specks inside the stack band,
    # neighboring-page print), so junk speckle cannot stall the walk.
    glyph_ink = _glyph_ink_mask(gray, junk_mask, ink_keep=ink_keep)
    glyph_cols = glyph_ink.any(axis=0)

    def strip_width(vals, glyphs, max_cols) -> int:
        """Width of the anchored depressed band (anchor boundary at index 0)."""
        end = 0
        gap_run = 0
        started = False
        for j in range(min(cap, max_cols, len(vals))):
            if glyphs[j]:
                break  # column carries text glyphs -> never cross it
            if vals[j] < rec:
                end = j + 1
                gap_run = 0
                started = True
            else:
                if not started:
                    if j >= anchor:
                        break
                else:
                    gap_run += 1
                    if gap_run > gap:
                        break
        return end

    def strip_is_junk(x0: int, x1: int) -> bool:
        """True when the strip has a continuous non-white vertical run that
        text can never produce (characters always leave gaps)."""
        if x1 <= x0:
            return False
        seg = (central[:, x0:x1] < rec).astype(np.uint8)
        # bridge small speckle holes so a textured/faint band still counts
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))
        run = max(int(h * 0.08), 1)
        return bool(cv2.erode(seg, np.ones((run, 1), np.uint8)).any())

    out = image.copy()

    def paint(x0: int, x1: int):
        """Whiten columns [x0, x1) full height, but keep glyph-sized ink
        (page numbers / running heads crossing the strip); tall ink and ink
        on border-connected junk (stack-band speckle) is painted with the
        rest."""
        region = out[:, x0:x1]
        region[~glyph_ink[:, x0:x1]] = paper_value
        out[:, x0:x1] = region

    # The strip may only occupy the outer edge zone of the image: shadows
    # hug the page edge, content does not.
    li = strip_width(cs[anchor_l:], glyph_cols[anchor_l:],
                     max(0, edge_zone - anchor_l))
    if li > 0 and strip_is_junk(anchor_l, anchor_l + li):
        paint(0, anchor_l + li)
    else:
        li = 0
    ri = strip_width(cs[:anchor_r][::-1], glyph_cols[:anchor_r][::-1],
                     max(0, anchor_r - (w - edge_zone)))
    if ri > 0 and strip_is_junk(anchor_r - ri, anchor_r):
        paint(anchor_r - ri, w)
    else:
        ri = 0
    if debug_out is not None:
        debug_out['edge_band_l'] = li
        debug_out['edge_band_r'] = ri
    return out


def _whiten_shadow_smudges(
    image: np.ndarray,
    paper_value=255,
    dark_ratio: float = 0.93,
    sharp_lap: float = 25.0,
    sharp_max_frac: float = 0.50,
    min_area: int = 800,
    side_zone_frac: float = 0.14,
    head_foot_frac: float = 0.10,
    junk_mask: Optional[np.ndarray] = None,
    debug_out: Optional[dict] = None,
) -> np.ndarray:
    """Paint out smooth dark smudges in the outer side margins.

    Targets flat-field lens-shaped spine/gutter shadow residue: the flat-field
    turns a thin smooth binding shadow into wider lens-shaped smudges (plus
    faint horizontal streaks) that carry dark cores. They are dark but SMOOTH -
    only a small fraction of their dark pixels are sharp edges - whereas real
    text is mostly sharp strokes. Each dark connected component in the outer
    side_zone_frac (within the central height band) is painted out only when
    its sharp-pixel fraction is below sharp_max_frac, so the whole 2-D smudge
    is covered while sharp edge text (a title reaching near the margin, a body
    column) is kept. The side-zone / central-band limits also protect page
    numbers, running heads and body text.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape
    paper = float(np.median(gray[int(h * 0.25):int(h * 0.75),
                                 int(w * 0.35):int(w * 0.65)]))
    hf = int(h * head_foot_frac)
    side = int(w * side_zone_frac)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    sharp = (lap > sharp_lap)

    # Dark connected components, restricted to the central height band
    dark = (gray < max(60.0, paper * dark_ratio)).astype(np.uint8)
    dark[:hf] = 0
    dark[h - hf:] = 0
    # Close small gaps so lens core + surrounding streaks form one component
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    num, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)

    # Glyph-sized ink must never be painted: a running head or page number
    # can merge (via the closing above) with surrounding show-through into
    # one smudge blob whose sharpness is diluted below the gate. Tall ink
    # (lens-shadow cores) and ink on border-connected junk (neighboring-page
    # print, stack-band speckle) are not protected and are painted with the
    # smudge.
    protect = _glyph_ink_mask(gray, junk_mask)

    out = image.copy()
    n_painted = 0
    px_painted = 0
    for i in range(1, num):
        x = st[i, cv2.CC_STAT_LEFT]
        bw = st[i, cv2.CC_STAT_WIDTH]
        area = st[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx = x + bw // 2
        if not (cx < side or cx > w - side):
            continue  # not in an outer side margin -> leave it
        comp = lab == i
        if float(sharp[comp].mean()) >= sharp_max_frac:
            continue  # mostly sharp -> real text, keep it
        out[comp & ~protect] = paper_value
        n_painted += 1
        px_painted += int(area)
    if debug_out is not None:
        debug_out['smudges'] = n_painted
        debug_out['smudge_px'] = px_painted
    return out


# =============================================================================
# (B) Color Statistics and Adjustment
# =============================================================================
def calculate_color_stats(image: np.ndarray) -> ColorStats:
    """
    Calculate color statistics for a page.

    Exactly matches C# CalculateColorStats:
    1. Build luminance histogram (1/16 sampling)
    2. Find 5th percentile (ink) and 95th percentile (paper)
    3. Calculate average RGB for ink and paper regions

    Args:
        image: Input image (RGB or grayscale)

    Returns:
        ColorStats with paper and ink colors
    """
    if len(image.shape) == 2:
        # Convert grayscale to RGB for consistent processing
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Step 1: Sample image (every SAMPLE_STEP pixels) - VECTORIZED
    sampled = image[::SAMPLE_STEP, ::SAMPLE_STEP]  # (H', W', 3)

    # Calculate luminance for all sampled pixels using BT.601
    # lum = 0.299*R + 0.587*G + 0.114*B
    lum = (0.299 * sampled[:, :, 0] +
           0.587 * sampled[:, :, 1] +
           0.114 * sampled[:, :, 2] + 0.5).astype(np.int32)
    lum = np.clip(lum, 0, 255)

    # Build histogram
    hist = np.bincount(lum.ravel(), minlength=256)
    total = lum.size

    # Step 2: Find 5th and 95th percentile
    low_target = int(total * 0.05)
    high_target = int(total * 0.95)

    cumsum = np.cumsum(hist)

    # Find where cumsum crosses thresholds
    low_lum = np.searchsorted(cumsum, low_target, side='left')
    high_lum = np.searchsorted(cumsum, high_target, side='left')

    # Clamp to valid range
    low_lum = int(np.clip(low_lum, 0, 255))
    high_lum = int(np.clip(high_lum, 0, 255))

    # Step 3: Calculate average RGB for paper (>=high_lum) and ink (<=low_lum) - VECTORIZED
    lum_flat = lum.ravel()
    sampled_flat = sampled.reshape(-1, 3)  # (N, 3)

    paper_mask = lum_flat >= high_lum
    ink_mask = lum_flat <= low_lum

    paper_pixels = sampled_flat[paper_mask]
    ink_pixels = sampled_flat[ink_mask]

    # Calculate means
    if len(paper_pixels) > 0:
        paper_mean = paper_pixels.mean(axis=0)
    else:
        paper_mean = np.array([255.0, 255.0, 255.0])

    if len(ink_pixels) > 0:
        ink_mean = ink_pixels.mean(axis=0)
    else:
        ink_mean = np.array([0.0, 0.0, 0.0])

    return ColorStats(
        paper_r=float(paper_mean[0]),
        paper_g=float(paper_mean[1]),
        paper_b=float(paper_mean[2]),
        ink_r=float(ink_mean[0]),
        ink_g=float(ink_mean[1]),
        ink_b=float(ink_mean[2]),
    )


def percentile(values: List[float], p: float) -> float:
    """
    Calculate percentile with linear interpolation.
    Matches C# Percentile function.

    Args:
        values: List of values (will be sorted)
        p: Percentile (0-100)

    Returns:
        Percentile value
    """
    if len(values) == 0:
        return 0.0

    sorted_vals = sorted(values)
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))

    if lo == hi:
        return sorted_vals[lo]

    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def decide_global_color_adjustment(stats_list: List[ColorStats]) -> GlobalColorParam:
    """
    Decide global color adjustment parameters from multiple pages.

    Exactly matches C# DecideGlobalColorAdjustment:
    1. Outlier exclusion using median + MAD
    2. Calculate linear scale/offset per channel (ink→0, paper→255)
    3. Calculate ghost suppression threshold

    Args:
        stats_list: List of ColorStats from all pages

    Returns:
        GlobalColorParam with adjustment parameters
    """
    if len(stats_list) == 0:
        return GlobalColorParam()

    # Step 1: Outlier exclusion on paper color
    paper_y = [0.299 * s.paper_r + 0.587 * s.paper_g + 0.114 * s.paper_b
               for s in stats_list]
    med_y = percentile(paper_y, 50)
    mad = percentile([abs(v - med_y) for v in paper_y], 50)
    thr = mad * 1.5

    main_pages = [s for s, py in zip(stats_list, paper_y)
                  if abs(py - med_y) <= thr]
    if len(main_pages) == 0:
        main_pages = stats_list

    # Step 2: Calculate channel-specific median for paper and ink
    bg_r = percentile([s.paper_r for s in main_pages], 50)
    bg_g = percentile([s.paper_g for s in main_pages], 50)
    bg_b = percentile([s.paper_b for s in main_pages], 50)

    ink_r = percentile([s.ink_r for s in main_pages], 50)
    ink_g = percentile([s.ink_g for s in main_pages], 50)
    ink_b = percentile([s.ink_b for s in main_pages], 50)

    # Step 3: Calculate linear mapping (ink→0, paper→255)
    def calc_linear(bg: float, ink: float) -> Tuple[float, float]:
        diff = bg - ink
        if diff < 1:
            return (1.0, 0.0)
        scale = np.clip(255.0 / diff, SCALE_CLAMP_MIN, SCALE_CLAMP_MAX)
        offset = -ink * scale
        return (scale, offset)

    s_r, o_r = calc_linear(bg_r, ink_r)
    s_g, o_g = calc_linear(bg_g, ink_g)
    s_b, o_b = calc_linear(bg_b, ink_b)

    # Step 4: Calculate ghost suppression threshold
    def sc_clamp(v: float) -> int:
        return int(np.clip(v, 0, 255))

    bg_lum_scaled = (
        0.299 * sc_clamp(bg_r * s_r + o_r) +
        0.587 * sc_clamp(bg_g * s_g + o_g) +
        0.114 * sc_clamp(bg_b * s_b + o_b)
    )

    ink_lum_scaled = (
        0.299 * sc_clamp(ink_r * s_r + o_r) +
        0.587 * sc_clamp(ink_g * s_g + o_g) +
        0.114 * sc_clamp(ink_b * s_b + o_b)
    )

    ghost_thr = int(np.clip((ink_lum_scaled + bg_lum_scaled) * 0.5, 0, 255))

    return GlobalColorParam(
        scale_r=s_r,
        scale_g=s_g,
        scale_b=s_b,
        offset_r=o_r,
        offset_g=o_g,
        offset_b=o_b,
        ghost_suppress_lum_threshold=ghost_thr,
        white_clip_range=WHITE_CLIP_RANGE,
        paper_r=int(round(bg_r)),
        paper_g=int(round(bg_g)),
        paper_b=int(round(bg_b)),
        sat_threshold=SAT_THRESHOLD,
        color_dist_threshold=COLOR_DIST_THRESHOLD,
    )


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    RGB to HSV conversion.
    Matches C# RgbToHsv (h: 0-360°, s/v: 0-1).
    """
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    max_c = max(rf, gf, bf)
    min_c = min(rf, gf, bf)
    v = max_c
    d = max_c - min_c
    s = 0.0 if max_c == 0 else d / max_c

    if d == 0:
        h = 0.0
    elif max_c == rf:
        h = 60.0 * (((gf - bf) / d) % 6.0)
    elif max_c == gf:
        h = 60.0 * (((bf - rf) / d) + 2.0)
    else:
        h = 60.0 * (((rf - gf) / d) + 4.0)

    if h < 0:
        h += 360.0

    return h, s, v


def apply_global_color_adjustment(image: np.ndarray, param: GlobalColorParam) -> np.ndarray:
    """
    Apply global color adjustment to image.

    Exactly matches C# ApplyGlobalColorAdjustment:
    1. Linear per-channel scaling
    2. Smooth-step whitening for near-paper colors
    3. Pastel pink/orange bleed removal

    Args:
        image: Input image (RGB)
        param: GlobalColorParam with adjustment parameters

    Returns:
        Adjusted image
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]

    clip_start = param.ghost_suppress_lum_threshold
    clip_end = max(0, min(255, 255 - param.white_clip_range))

    for y in range(h):
        for x in range(w):
            r, g, b = result[y, x]

            # Step 1: Linear scaling
            r = np.clip(r * param.scale_r + param.offset_r, 0, 255)
            g = np.clip(g * param.scale_g + param.offset_g, 0, 255)
            b = np.clip(b * param.scale_b + param.offset_b, 0, 255)

            # Step 2: Smooth-step whitening for near-paper colors
            lum = int((r * 299 + g * 587 + b * 114) / 1000)

            if lum >= clip_start:
                max_c = max(r, g, b)
                min_c = min(r, g, b)
                sat = 0 if max_c == 0 else int((max_c - min_c) * 255 / max_c)

                dist = (abs(int(r) - param.paper_r) +
                        abs(int(g) - param.paper_g) +
                        abs(int(b) - param.paper_b))

                if sat < param.sat_threshold and dist < param.color_dist_threshold:
                    t = np.clip((lum - clip_start) / (clip_end - clip_start + 1e-6), 0.0, 1.0)
                    wgt = t * t * (3.0 - 2.0 * t)  # Hermite smooth-step
                    r = np.clip(r + (255 - r) * wgt, 0, 255)
                    g = np.clip(g + (255 - g) * wgt, 0, 255)
                    b = np.clip(b + (255 - b) * wgt, 0, 255)

            # Step 3: Pastel pink removal (bleed suppression)
            hue, _, _ = rgb_to_hsv(int(r), int(g), int(b))
            max2 = max(r, g, b)
            min2 = min(r, g, b)
            sat2 = 0 if max2 == 0 else int((max2 - min2) * 255 / max2)
            lum2 = int((r * 299 + g * 587 + b * 114) / 1000)

            is_pastel_pink = (
                lum2 > 230 and
                sat2 < 30 and
                (hue <= 40.0 or hue >= 330.0)
            )

            if is_pastel_pink:
                r, g, b = 255, 255, 255

            result[y, x] = [r, g, b]

    return result.astype(np.uint8)


def apply_global_color_adjustment_fast(image: np.ndarray, param: GlobalColorParam) -> np.ndarray:
    """
    Optimized vectorized version of apply_global_color_adjustment.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    result = image.astype(np.float32)

    # Step 1: Linear scaling (vectorized)
    result[:, :, 0] = np.clip(result[:, :, 0] * param.scale_r + param.offset_r, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * param.scale_g + param.offset_g, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * param.scale_b + param.offset_b, 0, 255)

    clip_start = param.ghost_suppress_lum_threshold
    clip_end = max(0, min(255, 255 - param.white_clip_range))

    # Step 2: Calculate luminance
    lum = (result[:, :, 0] * 0.299 + result[:, :, 1] * 0.587 + result[:, :, 2] * 0.114).astype(np.int32)

    # Step 3: Calculate saturation and distance to paper color
    max_c = np.max(result, axis=2)
    min_c = np.min(result, axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(max_c > 0, (max_c - min_c) * 255 / max_c, 0).astype(np.int32)

    dist = (np.abs(result[:, :, 0] - param.paper_r) +
            np.abs(result[:, :, 1] - param.paper_g) +
            np.abs(result[:, :, 2] - param.paper_b)).astype(np.int32)

    # Step 4: Apply smooth-step whitening
    mask = (lum >= clip_start) & (sat < param.sat_threshold) & (dist < param.color_dist_threshold)

    t = np.clip((lum - clip_start) / (clip_end - clip_start + 1e-6), 0.0, 1.0)
    wgt = t * t * (3.0 - 2.0 * t)  # Hermite smooth-step

    for c in range(3):
        result[:, :, c] = np.where(
            mask,
            np.clip(result[:, :, c] + (255 - result[:, :, c]) * wgt, 0, 255),
            result[:, :, c]
        )

    # Step 5: Pastel pink removal (simplified - full HSV is expensive)
    # Only apply to very bright, very low saturation pixels in pink hue range
    bright_mask = lum > 230
    low_sat_mask = sat < 30

    # Approximate pink hue detection (R > G and R > B, or R dominant)
    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
    pink_mask = bright_mask & low_sat_mask & ((r > g) | (r > b))

    result[pink_mask] = [255, 255, 255]

    return result.astype(np.uint8)


# =============================================================================
# (C) Bounding Box Detection
# =============================================================================
def detect_text_bounding_box(image: np.ndarray,
                             border_px: Optional[int] = None) -> Tuple[int, int, int, int]:
    """
    Detect text bounding box in document image.

    Exactly matches C# DetectTextBoundingBox:
    1. Fill 1% border with white (ignore edge noise)
    2. Convert to grayscale
    3. Otsu threshold + invert
    4. Morphological opening (3x3)
    5. Find contours
    6. Filter tiny areas (<0.0025% of total)
    7. Return bounding rectangle of all valid contours

    Args:
        image: Input image (RGB or grayscale)
        border_px: Width of the border strip blanked to white before
            measuring. None keeps the C# default of 1% of each dimension.
            The blank strip caps the measurable extent, so text lying within
            it is invisible to the bbox; pass a small value (e.g. 2) when the
            image's margins were already cleaned (margin whitening), where
            text can legitimately reach near the page edge.

    Returns:
        Tuple (x, y, width, height) of bounding box, or (0,0,0,0) if none found
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Step 1: Fill border with white (255) to ignore edge noise
    if border_px is None:
        border_x = max(w // 100, 1)
        border_y = max(h // 100, 1)
    else:
        border_x = max(border_px, 1)
        border_y = max(border_px, 1)

    # Top
    gray[:border_y, :] = 255
    # Bottom
    gray[-border_y:, :] = 255
    # Left
    gray[:, :border_x] = 255
    # Right
    gray[:, -border_x:] = 255

    # Step 2: Otsu binarization (inverted - text becomes white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Morphological opening (3x3 kernel) to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 4: Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Filter tiny areas and collect bounding rects
    img_area = w * h
    min_area = max(int(img_area * BBOX_MIN_AREA_PERCENT), 10)

    rects = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw * rh >= min_area:
            rects.append((x, y, rw, rh))

    # Step 6: If no valid contours, return empty
    if len(rects) == 0:
        return (0, 0, 0, 0)

    # Step 7: Calculate bounding rectangle covering all valid contours
    min_x = min(r[0] for r in rects)
    min_y = min(r[1] for r in rects)
    max_x = max(r[0] + r[2] - 1 for r in rects)
    max_y = max(r[1] + r[3] - 1 for r in rects)

    return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


# =============================================================================
# (D) Group Crop Region Decision
# =============================================================================
def percentile_int(values: List[int], p: float) -> int:
    """
    Calculate percentile for integers with linear interpolation.
    Values must be pre-sorted in ascending order.
    """
    if len(values) == 0:
        return 0

    idx = p * (len(values) - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))

    if lo == hi:
        return values[lo]

    frac = idx - lo
    return int(round(values[lo] + (values[hi] - values[lo]) * frac))


def median_int(values: List[int]) -> int:
    """Calculate median of integers."""
    if len(values) == 0:
        return 0

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) // 2


def decide_group_crop_region(bounding_boxes: List[PageBoundingBox]) -> Tuple[int, int, int, int]:
    """
    Decide crop region for a group of pages.

    Exactly matches C# DecideGroupCropRegion:
    1. Validate bounding boxes (remove zero-area)
    2. Calculate IQR for each edge
    3. Mark outliers using Tukey fence (k=1.5)
    4. If inliers < 50%, use all pages
    5. Return median of inlier edges

    Args:
        bounding_boxes: List of PageBoundingBox for the group

    Returns:
        Tuple (x, y, width, height) of crop region
    """
    if not bounding_boxes:
        return (0, 0, 0, 0)

    # Step 1: Filter out zero-area boxes
    valid = [b for b in bounding_boxes if b.bbox[2] > 0 and b.bbox[3] > 0]
    if not valid:
        return (0, 0, 0, 0)

    # Extract edges (left, top, right, bottom)
    def get_edges(b: PageBoundingBox):
        x, y, w, h = b.bbox
        return (x, y, x + w - 1, y + h - 1)  # left, top, right, bottom

    edges = [get_edges(b) for b in valid]

    lefts = sorted([e[0] for e in edges])
    tops = sorted([e[1] for e in edges])
    rights = sorted([e[2] for e in edges])
    bottoms = sorted([e[3] for e in edges])

    # Step 2: Calculate quartiles and IQR
    q1_l = percentile_int(lefts, 0.25)
    q3_l = percentile_int(lefts, 0.75)
    iqr_l = max(q3_l - q1_l, 1)

    q1_t = percentile_int(tops, 0.25)
    q3_t = percentile_int(tops, 0.75)
    iqr_t = max(q3_t - q1_t, 1)

    q1_r = percentile_int(rights, 0.25)
    q3_r = percentile_int(rights, 0.75)
    iqr_r = max(q3_r - q1_r, 1)

    q1_b = percentile_int(bottoms, 0.25)
    q3_b = percentile_int(bottoms, 0.75)
    iqr_b = max(q3_b - q1_b, 1)

    # Step 3: Define outlier detection
    def is_outlier(v: int, q1: int, q3: int, iqr: int) -> bool:
        return v < q1 - TUKEY_K * iqr or v > q3 + TUKEY_K * iqr

    # Step 4: Filter outliers (all 4 edges must be within bounds)
    inliers = []
    for b, e in zip(valid, [get_edges(bb) for bb in valid]):
        left, top, right, bottom = e
        if not (is_outlier(left, q1_l, q3_l, iqr_l) or
                is_outlier(top, q1_t, q3_t, iqr_t) or
                is_outlier(right, q1_r, q3_r, iqr_r) or
                is_outlier(bottom, q1_b, q3_b, iqr_b)):
            inliers.append(b)

    # Step 5: If inliers are too few, use all valid pages
    if len(inliers) < max(3, len(valid) // 2):
        inliers = valid

    # Step 6: Calculate median of inlier edges
    inlier_edges = [get_edges(b) for b in inliers]

    left = median_int([e[0] for e in inlier_edges])
    top = median_int([e[1] for e in inlier_edges])
    right = median_int([e[2] for e in inlier_edges])
    bottom = median_int([e[3] for e in inlier_edges])

    width = max(right - left, 0)
    height = max(bottom - top, 0)

    if width == 0 or height == 0:
        return (0, 0, 0, 0)

    return (left, top, width, height)


def decide_group_crop_region_envelope(
    bounding_boxes: List[PageBoundingBox],
) -> Tuple[int, int, int, int]:
    """Decide a group crop region that never clips any inlier page's text.

    Same Tukey-fence outlier rejection as :func:`decide_group_crop_region`
    (so junk / figure pages whose text bbox blows out one edge are still
    excluded), but the final region is the *envelope* (min left, min top,
    max right, max bottom) of the inliers instead of their median.

    The median-based crop places its right edge at the median text extent, so
    every page whose text reaches the common (gutter-side) maximum has its
    outermost column sliced off - the reported "right-edge characters cut"
    bug. Because a shared crop must contain all pages' text to avoid clipping
    any of them, the correct boundary is the inlier extreme, not the median.
    Outliers are still dropped, so a single figure page cannot balloon the
    crop.

    Returns (left, top, width, height); (0, 0, 0, 0) when empty/degenerate.
    """
    if not bounding_boxes:
        return (0, 0, 0, 0)

    # Accept either the Cython PageBoundingBox (.Left/.Top/.Width/.Height, as
    # produced by the pipeline) or the Python one (.bbox tuple).
    def get_edges(b):
        if hasattr(b, 'Left'):
            x, y, w, h = b.Left, b.Top, b.Width, b.Height
        else:
            x, y, w, h = b.bbox
        return (x, y, w, h, x + w - 1, y + h - 1)  # x,y,w,h,right,bottom

    valid = [get_edges(b) for b in bounding_boxes]
    valid = [e for e in valid if e[2] > 0 and e[3] > 0]
    if not valid:
        return (0, 0, 0, 0)

    edges = [(e[0], e[1], e[4], e[5]) for e in valid]  # left, top, right, bottom
    lefts = sorted(e[0] for e in edges)
    tops = sorted(e[1] for e in edges)
    rights = sorted(e[2] for e in edges)
    bottoms = sorted(e[3] for e in edges)

    def fence(sorted_vals):
        q1 = percentile_int(sorted_vals, 0.25)
        q3 = percentile_int(sorted_vals, 0.75)
        iqr = max(q3 - q1, 1)
        return q1 - TUKEY_K * iqr, q3 + TUKEY_K * iqr

    fl, ft, fr, fb = fence(lefts), fence(tops), fence(rights), fence(bottoms)

    def is_out(v, f):
        return v < f[0] or v > f[1]

    inliers = []
    for e in edges:
        left, top, right, bottom = e
        if not (is_out(left, fl) or is_out(top, ft)
                or is_out(right, fr) or is_out(bottom, fb)):
            inliers.append(e)

    if len(inliers) < max(3, len(valid) // 2):
        inliers = edges

    left = min(e[0] for e in inliers)
    top = min(e[1] for e in inliers)
    right = max(e[2] for e in inliers)
    bottom = max(e[3] for e in inliers)

    width = max(right - left, 0)
    height = max(bottom - top, 0)
    if width == 0 or height == 0:
        return (0, 0, 0, 0)

    return (left, top, width, height)


def unify_crop_regions(
    odd_region: Tuple[int, int, int, int],
    even_region: Tuple[int, int, int, int],
    margin_percent: int,
    img_width: int,
    img_height: int,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Unify odd and even crop regions.

    Matches C# pipeline steps:
    1. Unify Y coordinates (use min top, max bottom)
    2. Equalize dimensions to max width/height
    3. Add margin based on content width
    4. Center-adjust smaller region
    5. Clamp to image boundaries

    Args:
        odd_region: (x, y, w, h) for odd pages
        even_region: (x, y, w, h) for even pages
        margin_percent: Margin percentage (default 10)
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (odd_region, even_region) after unification
    """
    ox, oy, ow, oh = odd_region
    ex, ey, ew, eh = even_region

    # Handle empty regions
    if ow == 0 or oh == 0:
        if ew == 0 or eh == 0:
            return ((0, 0, img_width, img_height), (0, 0, img_width, img_height))
        return ((ex, ey, ew, eh), (ex, ey, ew, eh))
    if ew == 0 or eh == 0:
        return ((ox, oy, ow, oh), (ox, oy, ow, oh))

    # Step 1: Unify Y coordinates
    total_top = min(oy, ey)
    odd_bottom = oy + oh
    even_bottom = ey + eh
    total_bottom = max(odd_bottom, even_bottom)

    # Update heights
    oh = total_bottom - total_top
    eh = total_bottom - total_top
    oy = total_top
    ey = total_top

    # Step 2: Find max dimensions
    max_width = max(ow, ew)
    max_height = max(oh, eh)

    # Step 3: Add margin (based on width)
    margin_pixels = max_width * margin_percent // 100
    max_width += margin_pixels
    max_height += margin_pixels

    # Step 4: Adjust odd region
    if ow < max_width or oh < max_height:
        dw = max_width - ow
        dh = max_height - oh
        new_left = ox - dw // 2
        new_top = oy - dh // 2

        # Clamp to boundaries
        max_width = min(max_width, img_width)
        new_left = max(0, min(new_left, img_width - max_width))
        max_height = min(max_height, img_height)
        new_top = max(0, min(new_top, img_height - max_height))

        ox, oy, ow, oh = new_left, new_top, max_width, max_height

    # Step 5: Adjust even region
    if ew < max_width or eh < max_height:
        dw = max_width - ew
        dh = max_height - eh
        new_left = ex - dw // 2
        new_top = ey - dh // 2

        # Clamp to boundaries
        max_width = min(max_width, img_width)
        new_left = max(0, min(new_left, img_width - max_width))
        max_height = min(max_height, img_height)
        new_top = max(0, min(new_top, img_height - max_height))

        ex, ey, ew, eh = new_left, new_top, max_width, max_height

    return ((ox, oy, ow, oh), (ex, ey, ew, eh))


# =============================================================================
# (E) Resize with Natural Paper Color Padding
# =============================================================================
def sample_corner_colors(
    image: np.ndarray,
    patch_percent: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample average colors from 4 corners of image.

    Args:
        image: Input RGB image
        patch_percent: Percentage of image for each corner patch

    Returns:
        Tuple of (top_left, top_right, bottom_left, bottom_right) RGB colors
    """
    h, w = image.shape[:2]
    patch_w = max(w * patch_percent // 100, 8)
    patch_h = max(h * patch_percent // 100, 8)

    def average_color(region: np.ndarray) -> np.ndarray:
        return region.mean(axis=(0, 1)).astype(np.uint8)

    tl = average_color(image[:patch_h, :patch_w])
    tr = average_color(image[:patch_h, -patch_w:])
    bl = average_color(image[-patch_h:, :patch_w])
    br = average_color(image[-patch_h:, -patch_w:])

    return tl, tr, bl, br


def bilinear_interpolate(
    tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray,
    u: float, v: float,
) -> np.ndarray:
    """Bilinear interpolation between 4 corner colors."""
    top = tl * (1 - u) + tr * u
    bottom = bl * (1 - u) + br * u
    return (top * (1 - v) + bottom * v).astype(np.uint8)


def resize_with_natural_paper_padding(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    shift_x: int = 0,
    shift_y: int = 0,
    corner_patch_percent: int = 3,
    feather: int = 4,
) -> np.ndarray:
    """
    Resize image and add natural paper color padding.

    Matches C# ResizeAndMakePaddingWithNaturalPaperColor2:
    1. Resize to fit target while maintaining aspect ratio
    2. Sample corner colors
    3. Create bilinear gradient background
    4. Place image with optional shift
    5. Apply feathering at edges

    Args:
        image: Input image (RGB)
        target_width: Target canvas width
        target_height: Target canvas height
        shift_x: X shift in source coordinates
        shift_y: Y shift in source coordinates
        corner_patch_percent: Percentage for corner sampling
        feather: Feather blend pixels

    Returns:
        Resized and padded image
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, w = image.shape[:2]

    # Step 1: Calculate scale to fit target
    scale = min(target_width / w, target_height / h)
    fitted_w = int(round(w * scale))
    fitted_h = int(round(h * scale))

    # Resize
    fitted = cv2.resize(image, (fitted_w, fitted_h), interpolation=cv2.INTER_LANCZOS4)

    # Step 2: Calculate offset with shift
    shift_x_scaled = int(round(shift_x * scale))
    shift_y_scaled = int(round(shift_y * scale))
    off_x = shift_x_scaled
    off_y = shift_y_scaled

    # Step 3: Sample corner colors from fitted image
    tl, tr, bl, br = sample_corner_colors(fitted, corner_patch_percent)

    # Step 4: Create gradient background - VECTORIZED
    # Create coordinate grids
    v_coords = np.linspace(0, 1, target_height).reshape(-1, 1, 1)  # (H, 1, 1)
    u_coords = np.linspace(0, 1, target_width).reshape(1, -1, 1)   # (1, W, 1)

    # Convert corners to float for interpolation
    tl_f = tl.astype(np.float32).reshape(1, 1, 3)
    tr_f = tr.astype(np.float32).reshape(1, 1, 3)
    bl_f = bl.astype(np.float32).reshape(1, 1, 3)
    br_f = br.astype(np.float32).reshape(1, 1, 3)

    # Bilinear interpolation: top = tl*(1-u) + tr*u, bottom = bl*(1-u) + br*u
    # result = top*(1-v) + bottom*v
    top = tl_f * (1 - u_coords) + tr_f * u_coords      # (1, W, 3)
    bottom = bl_f * (1 - u_coords) + br_f * u_coords   # (1, W, 3)
    canvas = (top * (1 - v_coords) + bottom * v_coords).astype(np.uint8)  # (H, W, 3)

    # Step 5: Place fitted image on canvas
    # Clamp to valid range
    src_x_start = max(0, -off_x)
    src_y_start = max(0, -off_y)
    dst_x_start = max(0, off_x)
    dst_y_start = max(0, off_y)

    copy_w = min(fitted_w - src_x_start, target_width - dst_x_start)
    copy_h = min(fitted_h - src_y_start, target_height - dst_y_start)

    if copy_w > 0 and copy_h > 0:
        canvas[dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w] = \
            fitted[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w]

    # Step 6: Apply feathering at seams - VECTORIZED
    if feather > 0 and copy_w > 0 and copy_h > 0:
        # Pre-compute x coordinates and u values for the image region
        x_start = dst_x_start
        x_end = min(dst_x_start + copy_w, target_width)
        x_indices = np.arange(x_start, x_end)
        u_vals = x_indices / (target_width - 1) if target_width > 1 else np.zeros_like(x_indices, dtype=np.float32)

        # Top edge feathering
        for i in range(feather):
            y_idx = dst_y_start + i
            if 0 <= y_idx < target_height:
                alpha = i / feather
                v_val = y_idx / (target_height - 1) if target_height > 1 else 0

                # Vectorized bilinear interpolation for entire row
                u_arr = u_vals.reshape(-1, 1)  # (W', 1)
                top_row = tl.astype(np.float32) * (1 - u_arr) + tr.astype(np.float32) * u_arr
                bottom_row = bl.astype(np.float32) * (1 - u_arr) + br.astype(np.float32) * u_arr
                bg = (top_row * (1 - v_val) + bottom_row * v_val)  # (W', 3)

                canvas[y_idx, x_start:x_end] = (
                    canvas[y_idx, x_start:x_end].astype(np.float32) * alpha +
                    bg * (1 - alpha)
                ).astype(np.uint8)

        # Bottom edge feathering
        for i in range(feather):
            y_idx = dst_y_start + copy_h - 1 - i
            if 0 <= y_idx < target_height:
                alpha = i / feather
                v_val = y_idx / (target_height - 1) if target_height > 1 else 0

                # Vectorized bilinear interpolation for entire row
                u_arr = u_vals.reshape(-1, 1)
                top_row = tl.astype(np.float32) * (1 - u_arr) + tr.astype(np.float32) * u_arr
                bottom_row = bl.astype(np.float32) * (1 - u_arr) + br.astype(np.float32) * u_arr
                bg = (top_row * (1 - v_val) + bottom_row * v_val)

                canvas[y_idx, x_start:x_end] = (
                    canvas[y_idx, x_start:x_end].astype(np.float32) * alpha +
                    bg * (1 - alpha)
                ).astype(np.uint8)

    return canvas


def detect_page_orientation(image: np.ndarray) -> str:
    """
    Detect if the page is primarily vertical (Japanese) or horizontal (Western) text.

    C# lines 4693-4722: IsPaperVerticalWriting_GetProbability

    Uses line scanning approach:
    1. Scan horizontally to detect row patterns (horizontal text has regular line gaps)
    2. Rotate 90° and scan to detect column patterns (vertical text has regular column gaps)
    3. Compare scores to determine writing direction
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Binarize with Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # C# line 4706: Horizontal score (for horizontal writing detection)
    horizontal_score = _compute_linear_score(binary)

    # C# lines 4709-4711: Rotate 90° and compute vertical score
    rotated = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
    vertical_score = _compute_linear_score(rotated)

    # C# lines 4714-4715: Compute probability
    total = horizontal_score + vertical_score + 1e-9
    vertical_probability = vertical_score / total

    # C# line 4708: threshold is 0.5
    return 'vertical' if vertical_probability >= 0.5 else 'horizontal'


def _compute_linear_score(img: np.ndarray) -> float:
    """
    C# lines 4727-4773+: ComputeLinearScore

    Scan image row by row, counting black pixel "intersections" (transitions).
    Higher score means more regular line structure.
    """
    height, width = img.shape[:2]

    # C# lines 4732-4734: Divide into 4 blocks to handle multi-column
    block_width = width // 4
    block_scores = []

    for blk in range(4):
        start_x = blk * block_width
        end_x = width if blk == 3 else start_x + block_width

        # Count intersections per row
        intersections_per_row = []
        zero_lines = 0

        for y in range(height):
            row = img[y, start_x:end_x]
            # Count black pixel clusters (transitions from white to black)
            intersects = 0
            in_black = False
            for x in range(len(row)):
                if row[x] > 127:  # Black (inverted binary, so 255 = black)
                    if not in_black:
                        intersects += 1
                        in_black = True
                else:
                    in_black = False

            intersections_per_row.append(intersects)
            if intersects == 0:
                zero_lines += 1

        # C# lines 4775-4790: Compute statistics
        if len(intersections_per_row) == 0:
            block_scores.append(0.0)
            continue

        arr = np.array(intersections_per_row, dtype=np.float64)
        mean_val = np.mean(arr)
        std_val = np.std(arr)

        if mean_val < 1e-9:
            block_scores.append(0.0)
            continue

        # Coefficient of variation (lower = more regular)
        cv = std_val / mean_val

        # C# lines 4792-4808: Score calculation
        # Higher zero_lines ratio and lower CV = better line structure
        zero_ratio = zero_lines / height
        regularity = 1.0 / (1.0 + cv) if cv > 0 else 1.0

        # Combined score
        score = mean_val * regularity * (0.5 + zero_ratio)
        block_scores.append(score)

    # Return average of block scores
    return sum(block_scores) / len(block_scores) if block_scores else 0.0
