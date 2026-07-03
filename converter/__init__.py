"""
PDF Converter - Exact Python port of C# DN_SuperBook_PDF_Converter

This is a faithful reimplementation of SuperPdfUtil.cs including:
- Temp directory structure for intermediate files
- Internal high-resolution processing (4960x7016)
- Final output at 3508px height
- Odd/even page grouping
- IQR-based outlier removal for crop regions
- Global color adjustment with ghost/bleed suppression
- Natural paper color padding with gradient
- OCR page number detection
"""

from .pipeline import convert_pdf, ConversionOptions, ConversionResult
from .pdf_reader import extract_pages, get_page_count, crop_margins
from .pdf_writer import build_pdf
from .enhancer import enhance_image, create_enhancer
from .image_processing import (
    # Core algorithms
    deskew,
    calculate_color_stats,
    decide_global_color_adjustment,
    apply_global_color_adjustment_fast,
    detect_text_bounding_box,
    decide_group_crop_region,
    unify_crop_regions,
    resize_with_natural_paper_padding,
    detect_page_orientation,
    # Data classes
    ColorStats,
    GlobalColorParam,
    PageBoundingBox,
    # Constants
    INTERNAL_HIGH_RES_WIDTH,
    INTERNAL_HIGH_RES_HEIGHT,
    FINAL_TARGET_HEIGHT,
)

__version__ = "1.0.0"
__all__ = [
    # Main API
    "convert_pdf",
    "ConversionOptions",
    "ConversionResult",
    # PDF I/O
    "extract_pages",
    "get_page_count",
    "crop_margins",
    "build_pdf",
    # Enhancement
    "enhance_image",
    "create_enhancer",
    # Image processing
    "deskew",
    "calculate_color_stats",
    "decide_global_color_adjustment",
    "apply_global_color_adjustment_fast",
    "detect_text_bounding_box",
    "decide_group_crop_region",
    "unify_crop_regions",
    "resize_with_natural_paper_padding",
    "detect_page_orientation",
    # Data classes
    "ColorStats",
    "GlobalColorParam",
    "PageBoundingBox",
    # Constants
    "INTERNAL_HIGH_RES_WIDTH",
    "INTERNAL_HIGH_RES_HEIGHT",
    "FINAL_TARGET_HEIGHT",
]
