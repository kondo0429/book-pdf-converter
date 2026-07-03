"""
PDF output generation using PyMuPDF.
"""

from pathlib import Path
from typing import List, Optional, Iterator
import io
import numpy as np
import fitz  # PyMuPDF
from PIL import Image


def numpy_to_png_bytes(image: np.ndarray) -> bytes:
    """Convert numpy array to PNG bytes."""
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode='L')
    else:
        pil_img = Image.fromarray(image, mode='RGB')

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return buffer.getvalue()


def numpy_to_jpeg_bytes(image: np.ndarray, quality: int = 70) -> bytes:
    """Convert numpy array to JPEG bytes.

    Args:
        image: Input image as numpy array
        quality: JPEG quality (0-100), default 70 to match C#

    Returns:
        JPEG encoded bytes
    """
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode='L')
    else:
        pil_img = Image.fromarray(image, mode='RGB')

    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    return buffer.getvalue()


def build_pdf(
    images: Iterator[np.ndarray] | List[np.ndarray],
    output_path: str | Path,
    dpi: int = 300,
    title: Optional[str] = None,
    author: Optional[str] = None,
    compress: bool = True,
    image_format: str = 'jpeg',
    jpeg_quality: int = 70,
    physical_page_start: Optional[int] = None,
    logical_page_start: Optional[int] = None,
) -> None:
    """
    Build a PDF from a sequence of images.

    Args:
        images: Iterator or list of numpy arrays (pages)
        output_path: Output PDF path
        dpi: Resolution of input images
        title: PDF metadata title
        author: PDF metadata author
        compress: Apply PDF compression
        image_format: Image format in PDF ('jpeg' or 'png'), default 'jpeg' to match C#
        jpeg_quality: JPEG quality (0-100), default 70 to match C#
        physical_page_start: Physical page number where logical numbering starts (1-indexed)
        logical_page_start: Logical page number at physical_page_start
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()

    # Set metadata
    if title or author:
        metadata = doc.metadata
        if title:
            metadata['title'] = title
        if author:
            metadata['author'] = author
        doc.set_metadata(metadata)

    for i, img in enumerate(images):
        h, w = img.shape[:2]

        # Calculate page size in points (72 points = 1 inch)
        page_width = w * 72 / dpi
        page_height = h * 72 / dpi

        # Create new page
        page = doc.new_page(width=page_width, height=page_height)

        # Convert image to bytes (JPEG default to match C#)
        if image_format.lower() == 'png':
            img_bytes = numpy_to_png_bytes(img)
        else:
            img_bytes = numpy_to_jpeg_bytes(img, quality=jpeg_quality)

        # Insert image
        rect = fitz.Rect(0, 0, page_width, page_height)
        page.insert_image(rect, stream=img_bytes)

    # Set page labels if specified (C# SetPdfPageLabelAsync equivalent)
    # This maps physical pages to logical page numbers for PDF viewers
    if physical_page_start is not None and logical_page_start is not None:
        # PyMuPDF page labels: list of dicts with keys:
        #   startpage: 0-indexed physical page number
        #   style: 'D'=decimal, 'r'=roman lower, 'R'=roman upper, 'a'=alpha lower, 'A'=alpha upper
        #   prefix: string prefix (optional)
        #   firstpagenum: starting number (optional, default 1)
        labels = []
        phys_idx = physical_page_start - 1  # Convert to 0-indexed
        if phys_idx > 0:
            # Pages before the labeled section start at 1
            labels.append({"startpage": 0, "style": "D", "prefix": "", "firstpagenum": 1})
        labels.append({"startpage": phys_idx, "style": "D", "prefix": "", "firstpagenum": logical_page_start})
        doc.set_page_labels(labels)

    # Save with compression
    if compress:
        doc.save(str(output_path), garbage=4, deflate=True)
    else:
        doc.save(str(output_path))

    doc.close()


def build_pdf_from_files(
    image_paths: List[str | Path],
    output_path: str | Path,
    dpi: int = 300,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> None:
    """
    Build a PDF from image files.

    Args:
        image_paths: List of image file paths
        output_path: Output PDF path
        dpi: Resolution of input images
        title: PDF metadata title
        author: PDF metadata author
    """
    import cv2

    def load_images():
        for path in image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Convert BGR to RGB if color
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield img

    build_pdf(load_images(), output_path, dpi, title, author)


def add_page_labels(
    pdf_path: str | Path,
    start_page: int = 0,
    start_number: int = 1,
    prefix: str = "",
    style: str = "D",  # D=decimal, r=roman lowercase, R=roman uppercase
) -> None:
    """
    Add page labels (logical page numbers) to a PDF.

    Args:
        pdf_path: Path to PDF file (modified in place)
        start_page: Physical page index to start labeling
        start_number: Starting page number
        prefix: Prefix for page labels
        style: Numbering style
    """
    doc = fitz.open(str(pdf_path))

    # Page labels are set via PDF catalog
    # This is a simplified version - full implementation would use PDF page labels dict
    # For now, we can set it via the outline/bookmarks if needed

    doc.save(str(pdf_path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()


def merge_pdfs(
    input_paths: List[str | Path],
    output_path: str | Path,
) -> None:
    """
    Merge multiple PDFs into one.

    Args:
        input_paths: List of input PDF paths
        output_path: Output PDF path
    """
    doc = fitz.open()

    for path in input_paths:
        src = fitz.open(str(path))
        doc.insert_pdf(src)
        src.close()

    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()
