"""
Image file I/O that tolerates non-ASCII paths.

cv2.imread and cv2.imwrite hand the path to the C runtime, which on Windows
encodes it in the active code page, so a path holding characters outside it
never reaches the codec. The failure is quiet: imread returns None and imwrite
returns False, neither raising. It bites in two places - an input or output
folder named in Japanese, and the temp tree, which lives under the user profile
and so carries any non-ASCII characters in the account name.

Python takes a path as text, so do the file access here and hand OpenCV only
the encoded bytes.
"""

import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def imread_any_path(path: str | Path,
                    flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image, whatever characters its path holds.

    Returns None when the file is missing, empty or undecodable - the same
    way cv2.imread reports a failed read.
    """
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_any_path(path: str | Path, image: np.ndarray,
                     params: Optional[Sequence[int]] = None) -> None:
    """Write an image, whatever characters its path holds.

    Raises IOError if the image cannot be encoded or the file cannot be
    written. cv2.imwrite only returns False there, and every caller in this
    package ignored it, which turned an unwritable output directory into a
    later "cannot read" further down the pipeline.
    """
    ext = os.path.splitext(os.fspath(path))[1]
    try:
        ok, buf = cv2.imencode(ext, image, list(params) if params else [])
    except cv2.error as e:
        raise IOError(f"Cannot encode image for {path}: {e}") from e
    if not ok:
        raise IOError(f"Cannot encode image for {path}")
    try:
        buf.tofile(os.fspath(path))
    except OSError as e:
        raise IOError(f"Cannot write image: {path}: {e}") from e
