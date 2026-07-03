"""
PDF page extraction using PyMuPDF.
"""

from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import fitz  # PyMuPDF


def extract_pages(
    pdf_path: str | Path,
    dpi: int = 300,
    start_page: int = 0,
    end_page: Optional[int] = None,
    grayscale: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Extract pages from a PDF as numpy arrays.

    Args:
        pdf_path: Path to input PDF
        dpi: Resolution for rendering (default: 300)
        start_page: First page to extract (0-indexed)
        end_page: Last page to extract (exclusive), None for all
        grayscale: Convert to grayscale if True

    Yields:
        Tuple of (page_number, image_array)
        - page_number: 0-indexed page number
        - image_array: numpy array (H, W) for grayscale or (H, W, 3) for RGB
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if end_page is None:
        end_page = total_pages
    end_page = min(end_page, total_pages)

    # Zoom factor for desired DPI (PDF default is 72 DPI)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    colorspace = fitz.csGRAY if grayscale else fitz.csRGB

    for page_num in range(start_page, end_page):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix, colorspace=colorspace)

        # Convert to numpy array
        if grayscale:
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        else:
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        yield page_num, img.copy()

    doc.close()


def get_page_count(pdf_path: str | Path) -> int:
    """Get the number of pages in a PDF."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def crop_margins(image: np.ndarray, margin_percent: float = 0.5) -> np.ndarray:
    """
    Crop a percentage from each edge of the image.

    This removes scan borders/artifacts that may appear at the edges.

    Args:
        image: Input image array
        margin_percent: Percentage to crop from each edge (default: 0.5%)

    Returns:
        Cropped image array
    """
    h, w = image.shape[:2]
    margin_h = int(h * margin_percent / 100)
    margin_w = int(w * margin_percent / 100)

    if margin_h > 0 and margin_w > 0:
        return image[margin_h:-margin_h, margin_w:-margin_w].copy()
    return image
