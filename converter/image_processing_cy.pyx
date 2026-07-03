# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython port of SuperPdfUtil.cs image processing functions.
Line-by-line faithful translation from C# to maintain correctness.

Source: SuperBookTools/Basic/SuperPdfUtil.cs
"""

import math
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil, round, fabs, fmax, fmin, sqrt, atan, M_PI
from libc.stdlib cimport malloc, free
from libc.string cimport memset

# Type definitions
ctypedef np.uint8_t uint8
ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.float32_t float32
ctypedef np.float64_t float64

# =============================================================================
# Constants (matching C#)
# =============================================================================
DEF SAMPLE_STEP = 4  # C# line 2173: const int SAMPLE_STEP = 4

# Internal resolution for processing (matching C#)
DEF INTERNAL_HIGH_RES_WIDTH = 4960
DEF INTERNAL_HIGH_RES_HEIGHT = 7016
DEF FINAL_TARGET_HEIGHT = 3508


# =============================================================================
# Deskew Angle Detection (ported from ImageMagick)
# =============================================================================
# This is a Cython port of ImageMagick's DeskewImage() angle detection algorithm.
# Source: MagickCore/shear.c (ImageMagick, Apache 2.0 License)
# The algorithm uses Radon transform variant to find dominant line angle.
# =============================================================================

# Bit count lookup table (population count for bytes 0-255)
cdef unsigned short _BITS[256]
cdef int _init_j, _init_c, _init_count
for _init_j in range(256):
    _init_c = _init_j
    _init_count = 0
    while _init_c:
        _init_count += _init_c & 1
        _init_c >>= 1
    _BITS[_init_j] = _init_count


# Column-major matrix accessors (matches ImageMagick's MatrixInfo)
cdef inline unsigned short _deskew_get_elem(
    unsigned short* matrix,
    Py_ssize_t rows,
    Py_ssize_t x,
    Py_ssize_t y
) noexcept nogil:
    return matrix[x * rows + y]


cdef inline void _deskew_set_elem(
    unsigned short* matrix,
    Py_ssize_t rows,
    Py_ssize_t x,
    Py_ssize_t y,
    unsigned short value
) noexcept nogil:
    matrix[x * rows + y] = value


cdef void _radon_projection(
    unsigned short* source,
    unsigned short* dest,
    Py_ssize_t width,
    Py_ssize_t height,
    Py_ssize_t sign,
    size_t* projection
) noexcept nogil:
    """
    Port of RadonProjection() from MagickCore/shear.c:216-319

    Computes Radon transform projections using recursive doubling.
    Uses ping-pong buffering between source and dest.
    """
    cdef:
        unsigned short* p = source
        unsigned short* q = dest
        unsigned short* swap_tmp
        Py_ssize_t step, x, i, y, y_start
        unsigned short element, neighbor
        size_t sum
        Py_ssize_t delta

    step = 1
    while step < width:
        x = 0
        while x < width:
            i = 0
            while i < step:
                # Loop 1: y in [0, height-i-1)
                # Both y+i and y+i+1 neighbor accesses are valid
                y = 0
                while y < height - i - 1:
                    element = _deskew_get_elem(p, height, x + i, y)
                    neighbor = _deskew_get_elem(p, height, x + i + step, y + i) + element
                    _deskew_set_elem(q, height, x + 2*i, y, neighbor)
                    neighbor = _deskew_get_elem(p, height, x + i + step, y + i + 1) + element
                    _deskew_set_elem(q, height, x + 2*i + 1, y, neighbor)
                    y += 1

                # Loop 2: y == height - i - 1 (single iteration)
                # Only y+i is valid; writes element (not neighbor) to second output
                y = height - i - 1
                if y >= 0:
                    element = _deskew_get_elem(p, height, x + i, y)
                    if y + i < height:
                        neighbor = _deskew_get_elem(p, height, x + i + step, y + i) + element
                        _deskew_set_elem(q, height, x + 2*i, y, neighbor)
                    _deskew_set_elem(q, height, x + 2*i + 1, y, element)

                # Loop 3: y in [height-i, height)
                # No valid neighbors; copies element to both outputs
                y = height - i if height - i > 0 else 0
                while y < height:
                    element = _deskew_get_elem(p, height, x + i, y)
                    _deskew_set_elem(q, height, x + 2*i, y, element)
                    _deskew_set_elem(q, height, x + 2*i + 1, y, element)
                    y += 1

                i += 1
            x += 2 * step

        # Swap buffers (exactly like original shear.c:284-286)
        swap_tmp = p
        p = q
        q = swap_tmp
        step *= 2

    # Final projection: sum of squared differences between adjacent rows
    x = 0
    while x < width:
        sum = 0
        y = 0
        while y < height - 1:
            element = _deskew_get_elem(p, height, x, y)
            neighbor = _deskew_get_elem(p, height, x, y + 1)
            delta = <Py_ssize_t>element - <Py_ssize_t>neighbor
            sum = sum + <size_t>(delta * delta)
            y += 1
        projection[width + sign * x - 1] = sum
        x += 1


cdef double _compute_deskew_angle_nogil(
    const unsigned char* image,
    Py_ssize_t rows,
    Py_ssize_t cols,
    double threshold
) noexcept nogil:
    """
    Core nogil implementation of deskew angle detection.
    Port of DeskewImage() from MagickCore/shear.c:557-620

    Parameters
    ----------
    image : pointer to row-major grayscale uint8 image data
    rows, cols : image dimensions
    threshold : pixel threshold (< threshold = foreground)

    Returns
    -------
    double : detected skew angle in degrees
    """
    cdef:
        Py_ssize_t width = 1
        Py_ssize_t x, y, i
        Py_ssize_t bit, byte_val
        Py_ssize_t skew = 0
        unsigned short value
        size_t max_proj = 0
        size_t matrix_size, proj_size
        unsigned short* source = NULL
        unsigned short* dest = NULL
        size_t* projection = NULL
        double degrees

    # Calculate width: next power of 2 >= (cols+7)/8
    while width < ((cols + 7) // 8):
        width *= 2

    # Allocate buffers
    matrix_size = width * rows * sizeof(unsigned short)
    proj_size = (2 * width - 1) * sizeof(size_t)

    source = <unsigned short*>malloc(matrix_size)
    dest = <unsigned short*>malloc(matrix_size)
    projection = <size_t*>malloc(proj_size)

    if source == NULL or dest == NULL or projection == NULL:
        free(source)
        free(dest)
        free(projection)
        return 0.0  # Error: allocation failed

    memset(source, 0, matrix_size)
    memset(dest, 0, matrix_size)
    memset(projection, 0, proj_size)

    # ===== Pass 1: right-to-left bit packing (shear.c:381-430) =====
    y = 0
    while y < rows:
        bit = 0
        byte_val = 0
        i = (cols + 7) // 8
        x = 0
        while x < cols:
            byte_val = (byte_val << 1) & 0xFF
            if image[y * cols + x] < threshold:
                byte_val = byte_val | 1
            bit += 1
            if bit == 8:
                i -= 1
                _deskew_set_elem(source, rows, i, y, _BITS[byte_val])
                bit = 0
                byte_val = 0
            x += 1
        if bit != 0:
            byte_val = (byte_val << (8 - bit)) & 0xFF
            i -= 1
            _deskew_set_elem(source, rows, i, y, _BITS[byte_val])
        y += 1

    # First Radon projection with sign=-1
    _radon_projection(source, dest, width, rows, -1, projection)

    # Reset source for second pass
    memset(source, 0, matrix_size)

    # ===== Pass 2: left-to-right bit packing (shear.c:438-487) =====
    y = 0
    while y < rows:
        bit = 0
        byte_val = 0
        i = 0
        x = 0
        while x < cols:
            byte_val = (byte_val << 1) & 0xFF
            if image[y * cols + x] < threshold:
                byte_val = byte_val | 1
            bit += 1
            if bit == 8:
                _deskew_set_elem(source, rows, i, y, _BITS[byte_val])
                i += 1
                bit = 0
                byte_val = 0
            x += 1
        if bit != 0:
            byte_val = (byte_val << (8 - bit)) & 0xFF
            _deskew_set_elem(source, rows, i, y, _BITS[byte_val])
        y += 1

    # Second Radon projection with sign=+1
    _radon_projection(source, dest, width, rows, 1, projection)

    # Find maximum projection (shear.c:604-613)
    i = 0
    while i < 2 * width - 1:
        if projection[i] > max_proj:
            skew = i - width + 1
            max_proj = projection[i]
        i += 1

    # Convert to degrees (shear.c:615)
    degrees = -atan(<double>skew / <double>width / 8.0) * 180.0 / M_PI

    # Cleanup
    free(source)
    free(dest)
    free(projection)

    return degrees


def GetDeskewAngle(image, double threshold_percent=0.4):
    """
    Get deskew angle for a grayscale image.

    This is a Cython port of ImageMagick's DeskewImage() angle detection
    algorithm from MagickCore/shear.c (Apache 2.0 License).

    Parameters
    ----------
    image : 2D numpy array (uint8, C-contiguous)
        Grayscale image with values 0-255
    threshold_percent : float, default 0.4 (40%)
        Threshold for binarization as percentage (0.0-1.0).
        Pixels below threshold are foreground.
        40% threshold = 0.4 * 255 = 102

    Returns
    -------
    float
        Detected skew angle in degrees.
        Positive = clockwise skew, Negative = counter-clockwise skew.
    """
    if image.ndim != 2:
        raise ValueError("Image must be 2D grayscale")

    cdef const unsigned char[:, ::1] img = np.ascontiguousarray(image, dtype=np.uint8)
    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t cols = img.shape[1]
    cdef double result
    cdef double actual_threshold = threshold_percent * 255.0

    with nogil:
        result = _compute_deskew_angle_nogil(&img[0, 0], rows, cols, actual_threshold)

    return result


# =============================================================================
# Data Classes (matching C# lines 2997-3046)
# =============================================================================

cdef class ColorStats:
    """
    C# lines 2997-3013: internal class ColorStats
    """
    cdef public double MeanR, MeanG, MeanB
    cdef public double PaperR, PaperG, PaperB
    cdef public double InkR, InkG, InkB
    cdef public int PageNumber

    def __init__(self):
        self.MeanR = 255.0
        self.MeanG = 255.0
        self.MeanB = 255.0
        self.PaperR = 255.0
        self.PaperG = 255.0
        self.PaperB = 255.0
        self.InkR = 0.0
        self.InkG = 0.0
        self.InkB = 0.0
        self.PageNumber = 0


cdef class GlobalColorParam:
    """
    C# lines 3015-3039: internal class GlobalColorParam
    """
    cdef public double OffsetR, OffsetG, OffsetB
    cdef public double ScaleR, ScaleG, ScaleB
    cdef public uint8 GhostSuppressLumThreshold
    cdef public uint8 WhiteClipRange
    cdef public uint8 PaperR, PaperG, PaperB
    cdef public uint8 SatThreshold
    cdef public uint8 ColorDistThreshold
    cdef public float BleedHueMin, BleedHueMax, BleedValueMin

    def __init__(self):
        self.OffsetR = 0.0
        self.OffsetG = 0.0
        self.OffsetB = 0.0
        self.ScaleR = 1.0
        self.ScaleG = 1.0
        self.ScaleB = 1.0
        self.GhostSuppressLumThreshold = 200
        self.WhiteClipRange = 30
        self.PaperR = 255
        self.PaperG = 255
        self.PaperB = 255
        self.SatThreshold = 55
        self.ColorDistThreshold = 35
        self.BleedHueMin = 20.0
        self.BleedHueMax = 65.0
        self.BleedValueMin = 0.35


cdef class PageBoundingBox:
    """
    C# lines 3041-3046: internal class PageBoundingBox
    """
    cdef public int PageNumber
    cdef public int Left, Top, Width, Height

    def __init__(self, int page_number=0, int left=0, int top=0, int width=0, int height=0):
        self.PageNumber = page_number
        self.Left = left
        self.Top = top
        self.Width = width
        self.Height = height

    @property
    def Right(self):
        return self.Left + self.Width

    @property
    def Bottom(self):
        return self.Top + self.Height


# =============================================================================
# Helper Functions
# =============================================================================

cdef inline uint8 Clamp8(double v) noexcept nogil:
    """C# line 2400: static byte Clamp8(double v)"""
    if v < 0:
        return 0
    elif v > 255:
        return 255
    else:
        return <uint8>v


cdef inline int Clamp(int v, int min_val, int max_val) noexcept nogil:
    """Math.Clamp equivalent"""
    if v < min_val:
        return min_val
    elif v > max_val:
        return max_val
    else:
        return v


cdef inline int MaxInt(int a, int b) noexcept nogil:
    return a if a > b else b


cdef inline int MinInt(int a, int b) noexcept nogil:
    return a if a < b else b


cdef inline int Max3(int a, int b, int c) noexcept nogil:
    cdef int m = a
    if b > m:
        m = b
    if c > m:
        m = c
    return m


cdef inline int Min3(int a, int b, int c) noexcept nogil:
    cdef int m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m


# =============================================================================
# (B-1) CalculateColorStats
# C# lines 2170-2257
# =============================================================================

def CalculateColorStats(uint8[:,:,:] image):
    """
    C# lines 2170-2257: private static ColorStats CalculateColorStats(Image<Rgba32> image)

    Calculate color statistics for a page.
    1. Build luminance histogram (1/16 sampling with SAMPLE_STEP=4)
    2. Find 5th percentile (ink) and 95th percentile (paper)
    3. Calculate average RGB for ink and paper regions
    """
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef int x, y, i
    cdef uint8 r, g, b
    cdef int y8  # luminance

    # C# line 2174: Memory<long> hist = new long[256];
    cdef long[256] hist
    cdef long total = 0

    # Initialize histogram
    for i in range(256):
        hist[i] = 0

    # C# lines 2183-2194: Build luminance histogram
    with nogil:
        for y in range(0, h, SAMPLE_STEP):
            for x in range(0, w, SAMPLE_STEP):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]
                # C# line 2189: int y8 = (int)(0.299 * p.R + 0.587 * p.G + 0.114 * p.B + 0.5);
                y8 = <int>(0.299 * r + 0.587 * g + 0.114 * b + 0.5)
                if y8 > 255:
                    y8 = 255
                if y8 < 0:
                    y8 = 0
                hist[y8] += 1
                total += 1

    # C# lines 2197-2207: Find 5th and 95th percentile
    cdef long lowTarget = <long>(total * 0.05)
    cdef long highTarget = <long>(total * 0.95)

    cdef int lowLum = 0
    cdef int highLum = 255
    cdef long acc = 0

    for i in range(256):
        acc += hist[i]
        # C# line 2205: if (acc >= lowTarget && lowLum == 0) lowLum = i;
        if acc >= lowTarget and lowLum == 0:
            lowLum = i
        # C# line 2206: if (acc >= highTarget) { highLum = i; break; }
        if acc >= highTarget:
            highLum = i
            break

    # C# lines 2210-2235: Calculate average RGB for paper and ink
    cdef long sumPaperR = 0, sumPaperG = 0, sumPaperB = 0, cntPaper = 0
    cdef long sumInkR = 0, sumInkG = 0, sumInkB = 0, cntInk = 0

    with nogil:
        for y in range(0, h, SAMPLE_STEP):
            for x in range(0, w, SAMPLE_STEP):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]
                y8 = <int>(0.299 * r + 0.587 * g + 0.114 * b + 0.5)

                # C# line 2223-2227: if (y8 >= highLum) // Paper
                if y8 >= highLum:
                    sumPaperR += r
                    sumPaperG += g
                    sumPaperB += b
                    cntPaper += 1
                # C# line 2228-2232: else if (y8 <= lowLum) // Ink
                elif y8 <= lowLum:
                    sumInkR += r
                    sumInkG += g
                    sumInkB += b
                    cntInk += 1

    # C# lines 2237-2239: Prevent division by zero
    if cntPaper == 0:
        cntPaper = 1
    if cntInk == 0:
        cntInk = 1

    # C# lines 2242-2256: Build and return ColorStats
    cdef ColorStats stats = ColorStats()
    stats.MeanR = sumPaperR / <double>cntPaper
    stats.MeanG = sumPaperG / <double>cntPaper
    stats.MeanB = sumPaperB / <double>cntPaper
    stats.PaperR = sumPaperR / <double>cntPaper
    stats.PaperG = sumPaperG / <double>cntPaper
    stats.PaperB = sumPaperB / <double>cntPaper
    stats.InkR = sumInkR / <double>cntInk
    stats.InkG = sumInkG / <double>cntInk
    stats.InkB = sumInkB / <double>cntInk

    return stats


# =============================================================================
# (B-2) Percentile helper
# C# lines 2381-2390 (double version)
# =============================================================================

cdef double Percentile_double(list values, double p):
    """
    C# lines 2381-2390: private static double Percentile(List<double> list, double p)
    Simple percentile with linear interpolation.
    """
    if len(values) == 0:
        return 0.0

    values_sorted = sorted(values)
    cdef int n = len(values_sorted)
    cdef double rank = (p / 100.0) * (n - 1)
    cdef int lo = <int>floor(rank)
    cdef int hi = <int>ceil(rank)

    if lo == hi:
        return values_sorted[lo]

    return values_sorted[lo] + (values_sorted[hi] - values_sorted[lo]) * (rank - lo)


# =============================================================================
# (B-2.5) ExcludeOutliers
# C# lines 2208-2219
# =============================================================================

def ExcludeOutliers(list statsList):
    """
    C# lines 2208-2219: private static List<ColorStats> ExcludeOutliers(List<ColorStats> list)

    Remove top/bottom 20% of pages sorted by MeanR to exclude outlier pages.
    """
    cdef int n = len(statsList)
    cdef int skip, take

    # C# line 2210: if (list.Count < 3) return list
    if n < 3:
        return statsList

    # C# lines 2213-2217: Sort by MeanR, skip top/bottom 20%
    sorted_list = sorted(statsList, key=lambda s: s.MeanR)
    skip = <int>(n * 0.20)
    take = n - skip * 2
    if take < 1:
        take = 1

    # C# line 2217: sorted.Skip(skip).Take(take)
    return sorted_list[skip:skip + take]


# =============================================================================
# (B-3) DecideGlobalColorAdjustment
# C# lines 2271-2378
# =============================================================================

def DecideGlobalColorAdjustment(list statsList):
    """
    C# lines 2271-2378: private static GlobalColorParam DecideGlobalColorAdjustment(List<ColorStats> statsList)

    Decide global color adjustment parameters from a list of ColorStats.
    """
    cdef GlobalColorParam pOut = GlobalColorParam()

    # C# lines 2274-2285: Fallback for empty list
    if len(statsList) == 0:
        pOut.ScaleR = 1.0
        pOut.ScaleG = 1.0
        pOut.ScaleB = 1.0
        pOut.OffsetR = 0.0
        pOut.OffsetG = 0.0
        pOut.OffsetB = 0.0
        pOut.GhostSuppressLumThreshold = 200
        pOut.WhiteClipRange = 30
        return pOut

    # C# lines 2290-2298: Page outlier removal using median + MAD
    cdef list paperY = []
    cdef ColorStats s
    for s in statsList:
        paperY.append(0.299 * s.PaperR + 0.587 * s.PaperG + 0.114 * s.PaperB)

    cdef double medY = Percentile_double(paperY, 50.0)

    cdef list absDevs = []
    for v in paperY:
        absDevs.append(fabs(v - medY))
    cdef double mad = Percentile_double(absDevs, 50.0)
    cdef double thr = mad * 1.5

    # C# lines 2296-2299: Filter to main pages
    cdef list mainPages = []
    cdef double lumY
    for s in statsList:
        lumY = 0.299 * s.PaperR + 0.587 * s.PaperG + 0.114 * s.PaperB
        if fabs(lumY - medY) <= thr:
            mainPages.append(s)

    if len(mainPages) == 0:
        mainPages = statsList

    # C# lines 2304-2310: Channel-wise median for paper and ink
    cdef list paperRList = []
    cdef list paperGList = []
    cdef list paperBList = []
    cdef list inkRList = []
    cdef list inkGList = []
    cdef list inkBList = []

    for s in mainPages:
        paperRList.append(s.PaperR)
        paperGList.append(s.PaperG)
        paperBList.append(s.PaperB)
        inkRList.append(s.InkR)
        inkGList.append(s.InkG)
        inkBList.append(s.InkB)

    cdef double bgR = Percentile_double(paperRList, 50.0)
    cdef double bgG = Percentile_double(paperGList, 50.0)
    cdef double bgB = Percentile_double(paperBList, 50.0)
    cdef double inkR = Percentile_double(inkRList, 50.0)
    cdef double inkG = Percentile_double(inkGList, 50.0)
    cdef double inkB = Percentile_double(inkBList, 50.0)

    # C# lines 2317-2324: Linear scale calculation (ink->0, paper->255)
    cdef double sR, oR, sG, oG, sB, oB
    cdef double diff

    # For R channel
    diff = bgR - inkR
    if diff < 1:
        sR = 1.0
        oR = 0.0
    else:
        sR = 255.0 / diff
        if sR < 0.8:
            sR = 0.8
        elif sR > 4.0:
            sR = 4.0
        oR = -inkR * sR

    # For G channel
    diff = bgG - inkG
    if diff < 1:
        sG = 1.0
        oG = 0.0
    else:
        sG = 255.0 / diff
        if sG < 0.8:
            sG = 0.8
        elif sG > 4.0:
            sG = 4.0
        oG = -inkG * sG

    # For B channel
    diff = bgB - inkB
    if diff < 1:
        sB = 1.0
        oB = 0.0
    else:
        sB = 255.0 / diff
        if sB < 0.8:
            sB = 0.8
        elif sB > 4.0:
            sB = 4.0
        oB = -inkB * sB

    # C# lines 2334-2347: Calculate ghost suppression threshold
    cdef double bgLumScaled, inkLumScaled
    cdef double tmpR, tmpG, tmpB

    tmpR = bgR * sR + oR
    if tmpR < 0: tmpR = 0
    if tmpR > 255: tmpR = 255
    tmpG = bgG * sG + oG
    if tmpG < 0: tmpG = 0
    if tmpG > 255: tmpG = 255
    tmpB = bgB * sB + oB
    if tmpB < 0: tmpB = 0
    if tmpB > 255: tmpB = 255
    bgLumScaled = 0.299 * tmpR + 0.587 * tmpG + 0.114 * tmpB

    tmpR = inkR * sR + oR
    if tmpR < 0: tmpR = 0
    if tmpR > 255: tmpR = 255
    tmpG = inkG * sG + oG
    if tmpG < 0: tmpG = 0
    if tmpG > 255: tmpG = 255
    tmpB = inkB * sB + oB
    if tmpB < 0: tmpB = 0
    if tmpB > 255: tmpB = 255
    inkLumScaled = 0.299 * tmpR + 0.587 * tmpG + 0.114 * tmpB

    # C# line 2346-2347: Ghost threshold = midpoint between ink and paper
    cdef double ghostThr = (inkLumScaled + bgLumScaled) * 0.5
    if ghostThr < 0:
        ghostThr = 0
    if ghostThr > 255:
        ghostThr = 255

    # C# lines 2351-2376: Build output parameter
    pOut.ScaleR = sR
    pOut.OffsetR = oR
    pOut.ScaleG = sG
    pOut.OffsetG = oG
    pOut.ScaleB = sB
    pOut.OffsetB = oB
    pOut.GhostSuppressLumThreshold = <uint8>ghostThr
    pOut.WhiteClipRange = 30
    pOut.PaperR = <uint8>round(bgR)
    pOut.PaperG = <uint8>round(bgG)
    pOut.PaperB = <uint8>round(bgB)
    pOut.SatThreshold = 55
    pOut.ColorDistThreshold = 35
    pOut.BleedHueMin = 20.0
    pOut.BleedHueMax = 65.0
    pOut.BleedValueMin = 0.35

    return pOut


# =============================================================================
# (B-4) RgbToHsv
# C# lines 2468-2493
# =============================================================================

cdef void RgbToHsv(uint8 r, uint8 g, uint8 b, float* h, float* s, float* v) noexcept nogil:
    """
    C# lines 2468-2493: private static void RgbToHsv(byte r, byte g, byte b, out float h, out float s, out float v)
    RGB -> HSV conversion (h: 0-360, s/v: 0-1)
    """
    cdef float rf = r / 255.0
    cdef float gf = g / 255.0
    cdef float bf = b / 255.0

    cdef float max_val = rf
    if gf > max_val:
        max_val = gf
    if bf > max_val:
        max_val = bf

    cdef float min_val = rf
    if gf < min_val:
        min_val = gf
    if bf < min_val:
        min_val = bf

    v[0] = max_val
    cdef float d = max_val - min_val

    if max_val == 0:
        s[0] = 0
    else:
        s[0] = d / max_val

    if d == 0:
        h[0] = 0
    elif max_val == rf:
        h[0] = 60.0 * (((gf - bf) / d) % 6.0)
    elif max_val == gf:
        h[0] = 60.0 * (((bf - rf) / d) + 2.0)
    else:
        h[0] = 60.0 * (((rf - gf) / d) + 4.0)

    if h[0] < 0:
        h[0] += 360.0


# =============================================================================
# (B-5) ApplyGlobalColorAdjustment
# C# lines 2398-2463
# =============================================================================

def ApplyGlobalColorAdjustment(uint8[:,:,:] image, GlobalColorParam p):
    """
    C# lines 2398-2463: private static void ApplyGlobalColorAdjustment(Image<Rgba32> image, GlobalColorParam p)

    Apply global color adjustment to image (in-place).
    1. Linear correction (scale + offset)
    2. Smooth-step whitening for paper-like pixels
    3. Pastel pink removal
    """
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef int x, y

    cdef uint8 src_r, src_g, src_b
    cdef uint8 r, g, b
    cdef int lum, maxc, minc, sat, dist
    cdef double t, wgt
    cdef float hue, sat_f, val
    cdef int max2, min2, sat2, lum2
    cdef bint isPastelPink

    # C# line 2404-2405
    cdef uint8 paperR = p.PaperR
    cdef uint8 paperG = p.PaperG
    cdef uint8 paperB = p.PaperB
    cdef int clipStart = p.GhostSuppressLumThreshold
    cdef int clipEnd = Clamp(255 - p.WhiteClipRange, 0, 255)

    cdef double scaleR = p.ScaleR
    cdef double scaleG = p.ScaleG
    cdef double scaleB = p.ScaleB
    cdef double offsetR = p.OffsetR
    cdef double offsetG = p.OffsetG
    cdef double offsetB = p.OffsetB
    cdef uint8 satThreshold = p.SatThreshold
    cdef uint8 colorDistThreshold = p.ColorDistThreshold

    # C# lines 2409-2462
    with nogil:
        for y in range(h):
            for x in range(w):
                src_r = image[y, x, 0]
                src_g = image[y, x, 1]
                src_b = image[y, x, 2]

                # C# lines 2416-2418: Linear correction
                r = Clamp8(src_r * scaleR + offsetR)
                g = Clamp8(src_g * scaleG + offsetG)
                b = Clamp8(src_b * scaleB + offsetB)

                # C# lines 2421-2438: Smooth-step whitening for paper-like pixels
                lum = (r * 299 + g * 587 + b * 114) // 1000

                if lum >= clipStart:
                    maxc = Max3(r, g, b)
                    minc = Min3(r, g, b)
                    if maxc == 0:
                        sat = 0
                    else:
                        sat = (maxc - minc) * 255 // maxc

                    dist = abs(r - paperR) + abs(g - paperG) + abs(b - paperB)

                    # C# line 2431
                    if sat < satThreshold and dist < colorDistThreshold:
                        # C# lines 2433-2437: Hermite smooth-step
                        t = <double>(lum - clipStart) / (clipEnd - clipStart + 1e-6)
                        if t < 0:
                            t = 0
                        if t > 1:
                            t = 1
                        wgt = t * t * (3.0 - 2.0 * t)
                        r = Clamp8(r + (255 - r) * wgt)
                        g = Clamp8(g + (255 - g) * wgt)
                        b = Clamp8(b + (255 - b) * wgt)

                # C# lines 2441-2457: Pastel pink removal
                RgbToHsv(r, g, b, &hue, &sat_f, &val)
                max2 = Max3(r, g, b)
                min2 = Min3(r, g, b)
                if max2 == 0:
                    sat2 = 0
                else:
                    sat2 = (max2 - min2) * 255 // max2
                lum2 = (r * 299 + g * 587 + b * 114) // 1000

                # C# lines 2449-2452
                isPastelPink = (lum2 > 230 and sat2 < 30 and (hue <= 40.0 or hue >= 330.0))

                if isPastelPink:
                    r = 255
                    g = 255
                    b = 255

                # Write back
                image[y, x, 0] = r
                image[y, x, 1] = g
                image[y, x, 2] = b


# =============================================================================
# (D-1) Percentile for integers
# C# lines 2651-2662
# =============================================================================

cdef int Percentile_int(list values, double p):
    """
    C# lines 2651-2662: private static int Percentile(IReadOnlyList<int> values, double p)
    Percentile with linear interpolation for integers.
    Note: p is 0.0-1.0 in C# version (not 0-100)
    """
    if len(values) == 0:
        return 0

    # values should already be sorted in caller
    cdef int n = len(values)
    cdef double idx = p * (n - 1)
    cdef int lo = <int>floor(idx)
    cdef int hi = <int>ceil(idx)

    if lo == hi:
        return values[lo]

    cdef double frac = idx - lo
    return <int>round(values[lo] + (values[hi] - values[lo]) * frac)


# =============================================================================
# (D-2) Median
# C# lines 2667-2680
# =============================================================================

cdef int Median_int(list values):
    """
    C# lines 2667-2680: private static int Median(List<int> values)
    """
    if len(values) == 0:
        return 0

    values_sorted = sorted(values)
    cdef int n = len(values_sorted)

    if n % 2 == 1:
        return values_sorted[n // 2]
    else:
        return (values_sorted[n // 2 - 1] + values_sorted[n // 2]) // 2


# =============================================================================
# (D-3) DecideGroupCropRegion
# C# lines 2580-2645
# =============================================================================

def DecideGroupCropRegion(list boundingBoxes):
    """
    C# lines 2580-2645: private static Rectangle DecideGroupCropRegion(List<PageBoundingBox> boundingBoxes)

    Decide crop region for a group of pages using IQR outlier removal.
    """
    # C# lines 2583-2584
    if boundingBoxes is None or len(boundingBoxes) == 0:
        return (0, 0, 0, 0)  # (left, top, width, height)

    # C# lines 2587-2591: Filter out zero-area bounding boxes
    cdef list valid = []
    cdef PageBoundingBox bb
    for bb in boundingBoxes:
        if bb.Width > 0 and bb.Height > 0:
            valid.append(bb)

    if len(valid) == 0:
        return (0, 0, 0, 0)

    # C# lines 2595-2598: Collect and sort edge coordinates
    cdef list lefts = sorted([bb.Left for bb in valid])
    cdef list tops = sorted([bb.Top for bb in valid])
    cdef list rights = sorted([bb.Right for bb in valid])
    cdef list bottoms = sorted([bb.Bottom for bb in valid])

    # C# lines 2601-2604: Calculate quartiles and IQR
    cdef int q1L = Percentile_int(lefts, 0.25)
    cdef int q3L = Percentile_int(lefts, 0.75)
    cdef int iqrL = q3L - q1L

    cdef int q1T = Percentile_int(tops, 0.25)
    cdef int q3T = Percentile_int(tops, 0.75)
    cdef int iqrT = q3T - q1T

    cdef int q1R = Percentile_int(rights, 0.25)
    cdef int q3R = Percentile_int(rights, 0.75)
    cdef int iqrR = q3R - q1R

    cdef int q1B = Percentile_int(bottoms, 0.25)
    cdef int q3B = Percentile_int(bottoms, 0.75)
    cdef int iqrB = q3B - q1B

    # C# lines 2607-2610: Guard against IQR==0
    if iqrL == 0:
        iqrL = 1
    if iqrT == 0:
        iqrT = 1
    if iqrR == 0:
        iqrR = 1
    if iqrB == 0:
        iqrB = 1

    # C# lines 2613-2624: Filter outliers using Tukey fence (k=1.5)
    cdef double k = 1.5
    cdef list inliers = []
    cdef int left, top, right, bottom
    cdef bint isOutlier

    for bb in valid:
        left = bb.Left
        top = bb.Top
        right = bb.Right
        bottom = bb.Bottom

        # Check if any dimension is an outlier
        isOutlier = False
        if left < q1L - k * iqrL or left > q3L + k * iqrL:
            isOutlier = True
        if top < q1T - k * iqrT or top > q3T + k * iqrT:
            isOutlier = True
        if right < q1R - k * iqrR or right > q3R + k * iqrR:
            isOutlier = True
        if bottom < q1B - k * iqrB or bottom > q3B + k * iqrB:
            isOutlier = True

        if not isOutlier:
            inliers.append(bb)

    # C# lines 2627-2628: Fallback if too few inliers
    if len(inliers) < max(3, len(valid) // 2):
        inliers = valid

    # C# lines 2631-2634: Final region = median of inliers
    cdef int finalLeft = Median_int([bb.Left for bb in inliers])
    cdef int finalTop = Median_int([bb.Top for bb in inliers])
    cdef int finalRight = Median_int([bb.Right for bb in inliers])
    cdef int finalBottom = Median_int([bb.Bottom for bb in inliers])

    # C# lines 2637-2638: Calculate width and height
    cdef int w = finalRight - finalLeft
    cdef int h = finalBottom - finalTop

    if w < 0:
        w = 0
    if h < 0:
        h = 0

    # C# lines 2641-2642
    if w == 0 or h == 0:
        return (0, 0, 0, 0)

    return (finalLeft, finalTop, w, h)


# =============================================================================
# (E-0) EstimatePaperColor (fallback for AveragePaperColor)
# C# lines 1912-1987
# =============================================================================

cdef tuple EstimatePaperColor(np.ndarray[uint8, ndim=3] img):
    """
    C# lines 1912-1987: static Rgba32 EstimatePaperColor(Image<Rgba32> img)

    Estimates paper color from entire image using:
    1. Build 256-bin luminance histogram (2px stride sampling)
    2. Find top 5% luminance threshold
    3. Average low-saturation pixels (sat < 40) above threshold
    """
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    # C# lines 1917-1936: Build luminance histogram (2px stride)
    cdef int[256] lumHist
    cdef long samplePix = 0
    cdef int i, x, y
    cdef uint8 r, g, b
    cdef int lum

    for i in range(256):
        lumHist[i] = 0

    for y in range(0, h, 2):
        for x in range(0, w, 2):
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]
            lum = (r * 299 + g * 587 + b * 114) // 1000  # ITU-R BT.601
            lumHist[lum] += 1
            samplePix += 1

    # C# lines 1938-1950: Find top 5% luminance threshold
    cdef long target = <long>(samplePix * 0.05)
    cdef long acc = 0
    cdef int lumThreshold = 255

    for i in range(255, -1, -1):
        acc += lumHist[i]
        if acc >= target:
            lumThreshold = i
            break

    # C# lines 1952-1979: Average low-saturation pixels above threshold
    cdef long sumR = 0, sumG = 0, sumB = 0, cnt = 0
    cdef int max_c, min_c, sat

    for y in range(0, h, 2):
        for x in range(0, w, 2):
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]
            lum = (r * 299 + g * 587 + b * 114) // 1000

            if lum < lumThreshold:
                continue

            max_c = Max3(r, g, b)
            min_c = Min3(r, g, b)
            if max_c == 0:
                sat = 0
            else:
                sat = (max_c - min_c) * 255 // max_c  # 0-255

            # C# line 1970: Exclude high saturation pixels
            if sat < 40:
                sumR += r
                sumG += g
                sumB += b
                cnt += 1

    # C# line 1981: Fallback to white if no valid pixels
    if cnt == 0:
        return (255, 255, 255)

    return (<uint8>(sumR // cnt), <uint8>(sumG // cnt), <uint8>(sumB // cnt))


# =============================================================================
# (E-1) AveragePaperColor (helper for corner sampling)
# C# lines 2851-2928
# =============================================================================

cdef tuple AveragePaperColor(np.ndarray[uint8, ndim=3] img, int sx, int sy, int w, int h):
    """
    C# lines 2851-2928: private static Rgba32 AveragePaperColor(...)

    Calculate average paper color for a rectangular region.
    Uses luminance histogram to find top 5% brightest pixels.
    """
    cdef int imgH = img.shape[0]
    cdef int imgW = img.shape[1]

    # C# lines 2856-2859: Clip to image bounds
    if sx < 0:
        sx = 0
    if sx >= imgW:
        sx = imgW - 1
    if sy < 0:
        sy = 0
    if sy >= imgH:
        sy = imgH - 1

    if w < 1:
        w = 1
    if sx + w > imgW:
        w = imgW - sx
    if h < 1:
        h = 1
    if sy + h > imgH:
        h = imgH - sy

    # C# lines 2862-2880: Build luminance histogram (stride 2)
    cdef int[256] hist
    cdef int i, x, y
    cdef long samples = 0
    cdef uint8 r, g, b
    cdef int lum

    for i in range(256):
        hist[i] = 0

    for y in range(sy, sy + h, 2):
        for x in range(sx, sx + w, 2):
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]
            lum = (r * 299 + g * 587 + b * 114) // 1000
            hist[lum] += 1
            samples += 1

    if samples == 0:
        return EstimatePaperColor(img)  # C# line 2832-2833: fallback to full image estimate

    # C# lines 2886-2893: Find top 5% luminance threshold
    cdef long target = <long>(samples * 0.05)
    cdef long acc = 0
    cdef int thr = 255

    for i in range(255, -1, -1):
        acc += hist[i]
        if acc >= target:
            thr = i
            break

    # C# line 2896: Fallback if threshold too dark
    if thr < 150:
        return EstimatePaperColor(img)  # C# line 2846: fallback to full image estimate

    # C# lines 2899-2920: Average of low-saturation, bright pixels
    cdef long sumR = 0, sumG = 0, sumB = 0, cnt = 0
    cdef int max_c, min_c, sat

    for y in range(sy, sy + h, 2):
        for x in range(sx, sx + w, 2):
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]
            lum = (r * 299 + g * 587 + b * 114) // 1000

            if lum < thr:
                continue

            max_c = Max3(r, g, b)
            min_c = Min3(r, g, b)
            if max_c == 0:
                sat = 0
            else:
                sat = (max_c - min_c) * 255 // max_c

            # C# line 2915: Saturation filter
            if sat >= 40:
                continue

            sumR += r
            sumG += g
            sumB += b
            cnt += 1

    if cnt == 0:
        return EstimatePaperColor(img)  # C# line 2873: fallback to full image estimate

    return (<uint8>(sumR // cnt), <uint8>(sumG // cnt), <uint8>(sumB // cnt))


# =============================================================================
# (E-2) SampleCornerColors
# C# lines 2834-2848
# =============================================================================

def SampleCornerColors(np.ndarray[uint8, ndim=3] img, int percent=3):
    """
    C# lines 2834-2848: private static (Rgba32 tl, Rgba32 tr, Rgba32 bl, Rgba32 br) SampleCornerColors(...)

    Sample average colors from 4 corners of image.
    Returns tuple of (tl, tr, bl, br) where each is (r, g, b).
    """
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    # C# lines 2837-2838
    cdef int patchW = w * percent // 100
    cdef int patchH = h * percent // 100
    if patchW < 8:
        patchW = 8
    if patchH < 8:
        patchH = 8

    # C# lines 2840-2845
    tl = AveragePaperColor(img, 0, 0, patchW, patchH)
    tr = AveragePaperColor(img, w - patchW, 0, patchW, patchH)
    bl = AveragePaperColor(img, 0, h - patchH, patchW, patchH)
    br = AveragePaperColor(img, w - patchW, h - patchH, patchW, patchH)

    return (tl, tr, bl, br)


# =============================================================================
# (E-3) Bilinear interpolation
# C# lines 2931-2942
# =============================================================================

cdef inline void Bilinear_c(
    uint8 tlR, uint8 tlG, uint8 tlB,
    uint8 trR, uint8 trG, uint8 trB,
    uint8 blR, uint8 blG, uint8 blB,
    uint8 brR, uint8 brG, uint8 brB,
    float u, float v,
    uint8* outR, uint8* outG, uint8* outB
) noexcept nogil:
    """
    C# lines 2931-2942: private static Rgba32 Bilinear(...)
    Bilinear interpolation between 4 corner colors.
    u: 0=left, 1=right
    v: 0=top, 1=bottom

    Pure C version - no Python objects, no GIL.
    """
    # C# line 2937: static float Lerp(float a, float b, float t) => a + (b - a) * t;
    # Top row interpolation
    cdef float topR = tlR + (trR - tlR) * u
    cdef float topG = tlG + (trG - tlG) * u
    cdef float topB = tlB + (trB - tlB) * u

    # Bottom row interpolation
    cdef float botR = blR + (brR - blR) * u
    cdef float botG = blG + (brG - blG) * u
    cdef float botB = blB + (brB - blB) * u

    # Vertical interpolation
    outR[0] = <uint8>(topR + (botR - topR) * v)
    outG[0] = <uint8>(topG + (botG - topG) * v)
    outB[0] = <uint8>(topB + (botB - topB) * v)


# =============================================================================
# (E-4) ResizeAndMakePaddingWithNaturalPaperColor
# C# lines 2711-2765 (version 1) and 2774-2830 (version 2)
# =============================================================================

def ResizeAndMakePaddingWithNaturalPaperColor(
    np.ndarray[uint8, ndim=3] src,
    int targetW,
    int targetH,
    int cornerPatchPercent=3,
    int feather=4,
    int shiftX=0,
    int shiftY=0
):
    """
    C# lines 2711-2765: public static Image<Rgba32> ResizeAndMakePaddingWithNaturalPaperColor(...)

    Resize image to fit target while maintaining aspect ratio,
    then add natural paper color padding with gradient background.

    Note: This is a Python wrapper that uses PIL for resize (Lanczos3 to match C#).
    The core algorithm follows C# exactly.
    """
    from PIL import Image as PILImage

    cdef int srcH = src.shape[0]
    cdef int srcW = src.shape[1]

    # C# lines 2724-2727: Calculate scale to fit
    cdef double scale = min(<double>targetW / srcW, <double>targetH / srcH)
    cdef int fittedW = <int>round(srcW * scale)
    cdef int fittedH = <int>round(srcH * scale)

    # Resize using Lanczos3 (PIL's LANCZOS matches C#'s KnownResamplers.Lanczos3)
    # OpenCV only has INTER_LANCZOS4 (8x8 kernel), but C# uses Lanczos3 (6x6 kernel)
    pil_img = PILImage.fromarray(src)
    pil_resized = pil_img.resize((fittedW, fittedH), PILImage.LANCZOS)
    fitted_np = np.array(pil_resized)
    cdef uint8[:, :, :] fitted = fitted_np

    # C# lines 2731-2732: Calculate offset (centered + shift)
    cdef int offX = (targetW - fittedW) // 2 + <int>round(shiftX * scale)
    cdef int offY = (targetH - fittedH) // 2 + <int>round(shiftY * scale)

    # C# lines 2735-2736: Sample corner colors - extract to C variables
    corners = SampleCornerColors(fitted_np, cornerPatchPercent)
    cdef uint8 tlR = corners[0][0], tlG = corners[0][1], tlB = corners[0][2]
    cdef uint8 trR = corners[1][0], trG = corners[1][1], trB = corners[1][2]
    cdef uint8 blR = corners[2][0], blG = corners[2][1], blB = corners[2][2]
    cdef uint8 brR = corners[3][0], brG = corners[3][1], brB = corners[3][2]

    # C# lines 2739-2755: Create gradient background canvas
    canvas_np = np.zeros((targetH, targetW, 3), dtype=np.uint8)
    cdef uint8[:, :, :] canvas = canvas_np
    cdef int x, y
    cdef float ux, vy
    cdef uint8 outR, outG, outB
    cdef float targetHm1 = <float>(targetH - 1) if targetH > 1 else 1.0
    cdef float targetWm1 = <float>(targetW - 1) if targetW > 1 else 1.0

    # Release GIL for the gradient loop - pure C code
    with nogil:
        for y in range(targetH):
            vy = <float>y / targetHm1
            for x in range(targetW):
                ux = <float>x / targetWm1
                Bilinear_c(tlR, tlG, tlB, trR, trG, trB,
                          blR, blG, blB, brR, brG, brB,
                          ux, vy, &outR, &outG, &outB)
                canvas[y, x, 0] = outR
                canvas[y, x, 1] = outG
                canvas[y, x, 2] = outB

    # C# lines 2757-2759: Place fitted image on canvas
    cdef int srcXStart = max(0, -offX)
    cdef int srcYStart = max(0, -offY)
    cdef int dstXStart = max(0, offX)
    cdef int dstYStart = max(0, offY)
    cdef int copyW = min(fittedW - srcXStart, targetW - dstXStart)
    cdef int copyH = min(fittedH - srcYStart, targetH - dstYStart)

    cdef int cy, cx
    if copyW > 0 and copyH > 0:
        with nogil:
            for cy in range(copyH):
                for cx in range(copyW):
                    canvas[dstYStart + cy, dstXStart + cx, 0] = fitted[srcYStart + cy, srcXStart + cx, 0]
                    canvas[dstYStart + cy, dstXStart + cx, 1] = fitted[srcYStart + cy, srcXStart + cx, 1]
                    canvas[dstYStart + cy, dstXStart + cx, 2] = fitted[srcYStart + cy, srcXStart + cx, 2]

    # C# lines 2762: Apply feathering
    # Note: Our implementation differs from C# here (see CLAUDE.md exception #6)
    if feather > 0 and copyW > 0 and copyH > 0:
        _Feather_c(canvas, tlR, tlG, tlB, trR, trG, trB, blR, blG, blB, brR, brG, brB,
                   offX, offY, fittedW, fittedH, feather, targetW, targetH)

    return canvas_np


# =============================================================================
# (E-4b) ResizeAndMakePaddingWithNaturalPaperColor2
# C# lines 2774-2830
# =============================================================================

def ResizeAndMakePaddingWithNaturalPaperColor2(
    np.ndarray[uint8, ndim=3] src,
    int targetW,
    int targetH,
    int x,          # X shift in source coordinates (typically -cropRegion.Left)
    int y,          # Y shift in source coordinates (typically -cropRegion.Top)
    double scale,   # Scale factor (finalWidth / cropRegion.Width)
    int cornerPatchPercent=3,
    int feather=4
):
    """
    C# lines 2774-2830: ResizeAndMakePaddingWithNaturalPaperColor2(...)

    Similar to ResizeAndMakePaddingWithNaturalPaperColor but uses a fixed scale
    and applies shift to effectively crop the image.

    Args:
        src: Source image (RGB)
        targetW, targetH: Output canvas size
        x, y: Shift in source coordinates (negative of crop region left/top)
        scale: Scale factor to apply to source
        cornerPatchPercent: Percentage for corner sampling (default 3)
        feather: Feathering pixels (default 4)
    """
    from PIL import Image as PILImage

    cdef int srcH = src.shape[0]
    cdef int srcW = src.shape[1]

    # C# lines 2789-2790: Resize by fixed scale
    cdef int fittedW = <int>round(srcW * scale)
    cdef int fittedH = <int>round(srcH * scale)

    # Resize using Lanczos3 (PIL's LANCZOS matches C#'s KnownResamplers.Lanczos3)
    # OpenCV only has INTER_LANCZOS4 (8x8 kernel), but C# uses Lanczos3 (6x6 kernel)
    pil_img = PILImage.fromarray(src)
    pil_resized = pil_img.resize((fittedW, fittedH), PILImage.LANCZOS)
    fitted_np = np.array(pil_resized)
    cdef uint8[:, :, :] fitted = fitted_np

    # C# lines 2795-2796: Convert shift from source to fitted coordinates
    cdef int shiftX = <int>round(x * scale)
    cdef int shiftY = <int>round(y * scale)

    # C# lines 2799-2800: Offset is just the shift (no centering)
    cdef int offX = shiftX
    cdef int offY = shiftY

    # C# lines 2803: Sample corner colors from fitted image
    cdef tuple corners = SampleCornerColors(fitted_np, cornerPatchPercent)
    cdef tuple cTL = corners[0]
    cdef tuple cTR = corners[1]
    cdef tuple cBL = corners[2]
    cdef tuple cBR = corners[3]

    cdef uint8 tlR = cTL[0], tlG = cTL[1], tlB = cTL[2]
    cdef uint8 trR = cTR[0], trG = cTR[1], trB = cTR[2]
    cdef uint8 blR = cBL[0], blG = cBL[1], blB = cBL[2]
    cdef uint8 brR = cBR[0], brG = cBR[1], brB = cBR[2]

    # C# lines 2806-2819: Create canvas with gradient background
    canvas_np = np.zeros((targetH, targetW, 3), dtype=np.uint8)
    cdef uint8[:, :, :] canvas = canvas_np

    cdef int px, py
    cdef float ux, vy
    cdef float targetWm1 = <float>(targetW - 1) if targetW > 1 else 1.0
    cdef float targetHm1 = <float>(targetH - 1) if targetH > 1 else 1.0
    cdef uint8 outR, outG, outB

    # Gradient background
    with nogil:
        for py in range(targetH):
            vy = <float>py / targetHm1
            for px in range(targetW):
                ux = <float>px / targetWm1
                Bilinear_c(tlR, tlG, tlB, trR, trG, trB,
                          blR, blG, blB, brR, brG, brB,
                          ux, vy, &outR, &outG, &outB)
                canvas[py, px, 0] = outR
                canvas[py, px, 1] = outG
                canvas[py, px, 2] = outB

    # C# lines 2821-2824: Place fitted image (with clipping)
    cdef int srcXStart = max(0, -offX)
    cdef int srcYStart = max(0, -offY)
    cdef int dstXStart = max(0, offX)
    cdef int dstYStart = max(0, offY)
    cdef int copyW = min(fittedW - srcXStart, targetW - dstXStart)
    cdef int copyH = min(fittedH - srcYStart, targetH - dstYStart)

    cdef int cy, cx
    if copyW > 0 and copyH > 0:
        with nogil:
            for cy in range(copyH):
                for cx in range(copyW):
                    canvas[dstYStart + cy, dstXStart + cx, 0] = fitted[srcYStart + cy, srcXStart + cx, 0]
                    canvas[dstYStart + cy, dstXStart + cx, 1] = fitted[srcYStart + cy, srcXStart + cx, 1]
                    canvas[dstYStart + cy, dstXStart + cx, 2] = fitted[srcYStart + cy, srcXStart + cx, 2]

    # C# line 2827: Apply feathering
    # Note: Our implementation differs from C# here (see CLAUDE.md exception #6)
    if feather > 0 and copyW > 0 and copyH > 0:
        _Feather_c(canvas, tlR, tlG, tlB, trR, trG, trB, blR, blG, blB, brR, brG, brB,
                   offX, offY, fittedW, fittedH, feather, targetW, targetH)

    return canvas_np


# =============================================================================
# (E-5) Feather
# C# lines 2945-2984
# =============================================================================

cdef void _Feather_c(
    uint8[:, :, :] canvas,
    uint8 tlR, uint8 tlG, uint8 tlB,
    uint8 trR, uint8 trG, uint8 trB,
    uint8 blR, uint8 blG, uint8 blB,
    uint8 brR, uint8 brG, uint8 brB,
    int offX, int offY,
    int fittedW, int fittedH,
    int range_px,
    int canvasW, int canvasH
) noexcept:
    """
    C# lines 2945-2984: private static void Feather(...)
    Apply feathering at the seams between image and background.
    Pure C version.
    """
    cdef int x, y, dx, dy, d
    cdef float alpha, ux, vy
    cdef uint8 bgR, bgG, bgB, fgR, fgG, fgB
    cdef float canvasWm1 = <float>(canvasW - 1) if canvasW > 1 else 1.0
    cdef float canvasHm1 = <float>(canvasH - 1) if canvasH > 1 else 1.0
    cdef float range_px_f = <float>range_px

    with nogil:
        for y in range(offY - range_px, offY + fittedH + range_px):
            if y < 0 or y >= canvasH:
                continue

            for x in range(offX - range_px, offX + fittedW + range_px):
                if x < 0 or x >= canvasW:
                    continue

                # C# lines 2962-2964: Calculate distance to edge
                dx = offX - x
                if dx < 0:
                    dx = x - (offX + fittedW - 1)
                    if dx < 0:
                        dx = 0
                dy = offY - y
                if dy < 0:
                    dy = y - (offY + fittedH - 1)
                    if dy < 0:
                        dy = 0
                d = dx if dx > dy else dy

                # C# line 2965
                if d >= range_px:
                    continue

                # C# line 2967: alpha = 0 at boundary, 1 at full background
                alpha = <float>d / range_px_f

                # Get background color
                ux = <float>x / canvasWm1
                vy = <float>y / canvasHm1
                Bilinear_c(tlR, tlG, tlB, trR, trG, trB,
                          blR, blG, blB, brR, brG, brB,
                          ux, vy, &bgR, &bgG, &bgB)

                # Current pixel (foreground)
                fgR = canvas[y, x, 0]
                fgG = canvas[y, x, 1]
                fgB = canvas[y, x, 2]

                # Lerp blend: (1.0 - alpha) keeps foreground at center, fades to background at edges
                # Note: We use (1.0 - alpha) for visible blending; see CLAUDE.md exception #6
                canvas[y, x, 0] = <uint8>(bgR + (fgR - bgR) * (1.0 - alpha))
                canvas[y, x, 1] = <uint8>(bgG + (fgG - bgG) * (1.0 - alpha))
                canvas[y, x, 2] = <uint8>(bgB + (fgB - bgB) * (1.0 - alpha))


# =============================================================================
# (C) DetectTextBoundingBox
# C# lines 2508-2563
# =============================================================================

def DetectTextBoundingBox(np.ndarray[uint8, ndim=3] image):
    """
    C# lines 2508-2563: private static Rectangle DetectTextBoundingBox(...)

    Detect text bounding box in document image.
    1. Fill 1% border with white (ignore edge noise)
    2. Convert to grayscale
    3. Otsu threshold + invert
    4. Morphological opening (3x3)
    5. Find contours
    6. Filter tiny areas (<0.0025% of total)
    7. Return bounding rectangle of all valid contours

    Returns:
        Tuple (x, y, width, height) of bounding box, or (0,0,0,0) if none found
    """
    import cv2

    cdef int h = image.shape[0]
    cdef int w = image.shape[1]

    # C# line 2515: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # C# lines 2517-2527: Fill 1% border with white (255) to ignore edge noise
    cdef int borderX = w // 100
    cdef int borderY = h // 100
    if borderX < 1:
        borderX = 1
    if borderY < 1:
        borderY = 1

    # Top - C# line 2521
    cv2.rectangle(gray, (0, 0), (w, borderY), 255, -1)
    # Bottom - C# line 2523
    cv2.rectangle(gray, (0, h - borderY), (w, h), 255, -1)
    # Left - C# line 2525
    cv2.rectangle(gray, (0, 0), (borderX, h), 255, -1)
    # Right - C# line 2527
    cv2.rectangle(gray, (w - borderX, 0), (w, h), 255, -1)

    # C# line 2531: Otsu binarization (inverted - text becomes white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # C# lines 2534-2536: Morphological opening (3x3 kernel) to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # C# line 2539: Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # C# lines 2542-2550: Filter tiny areas and collect bounding rects
    cdef int imgArea = w * h
    cdef int minArea = <int>(imgArea * 0.000025)  # C# line 2543: 0.0025%
    if minArea < 10:
        minArea = 10

    rects = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw * rh >= minArea:
            rects.append((x, y, rw, rh))

    # C# lines 2553-2554: If no valid contours, return empty
    if len(rects) == 0:
        return (0, 0, 0, 0)

    # C# lines 2557-2560: Calculate bounding rectangle covering all valid contours
    cdef int minX = min(r[0] for r in rects)
    cdef int minY = min(r[1] for r in rects)
    cdef int maxX = max(r[0] + r[2] - 1 for r in rects)  # C# line 2559: r.X + r.Width - 1
    cdef int maxY = max(r[1] + r[3] - 1 for r in rects)  # C# line 2560: r.Y + r.Height - 1

    # C# line 2562: Return Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1)
    return (minX, minY, maxX - minX + 1, maxY - minY + 1)


# =============================================================================
# (D-4) AddMarginAndClip
# C# lines 2682-2699
# =============================================================================

def AddMarginAndClip(tuple rect, int margin_pixel, int img_width, int img_height):
    """
    C# lines 2682-2699: private static Rectangle AddMarginAndClip(...)

    Normalize crop region: clamp to image bounds and convert to inclusive width/height.

    Args:
        rect: (left, top, width, height) tuple
        margin_pixel: Margin to add (usually 0)
        img_width: Image width
        img_height: Image height

    Returns:
        (left, top, width, height) tuple after normalization
    """
    cdef int left = rect[0]
    cdef int top = rect[1]
    cdef int width = rect[2]
    cdef int height = rect[3]

    # C# lines 2684-2688: Handle empty regions
    if width <= 0 or height <= 0:
        return (0, 0, img_width, img_height)

    # C# lines 2690-2693: Calculate right/bottom (exclusive boundaries)
    # In C#, Rectangle.Right = Left + Width
    cdef int right = left + width
    cdef int bottom = top + height

    # Apply margin and clamp
    left = left - margin_pixel
    if left < 0:
        left = 0

    top = top - margin_pixel
    if top < 0:
        top = 0

    right = right + margin_pixel
    if right > img_width - 1:
        right = img_width - 1

    bottom = bottom + margin_pixel
    if bottom > img_height - 1:
        bottom = img_height - 1

    # C# lines 2695-2696: Calculate width/height with +1 (convert to inclusive)
    width = right - left + 1
    if width < 1:
        width = 1

    height = bottom - top + 1
    if height < 1:
        height = 1

    return (left, top, width, height)


# =============================================================================
# (D-5) UnifyCropRegions
# C# lines 1811-1871 (inline in PerformPagesYohakuAsync)
# =============================================================================

def UnifyCropRegions(
    tuple odd_region,
    tuple even_region,
    int margin_percent,
    int img_width,
    int img_height
):
    """
    C# lines 1764-1821: Unify odd and even crop regions.

    1. Unify Y coordinates (use min top, max bottom)
    2. Equalize dimensions to max width/height
    3. Add margin (percentage of each dimension)
    4. Center-adjust smaller region
    5. Clamp to image boundaries

    Args:
        odd_region: (x, y, w, h) for odd pages
        even_region: (x, y, w, h) for even pages
        margin_percent: Margin percentage (default 10)
        img_width: Image width (internalHighResImgWidth)
        img_height: Image height (internalHighResImgHeight)

    Returns:
        Tuple of (odd_region, even_region) after unification
    """
    cdef int ox = odd_region[0]
    cdef int oy = odd_region[1]
    cdef int ow = odd_region[2]
    cdef int oh = odd_region[3]

    cdef int ex = even_region[0]
    cdef int ey = even_region[1]
    cdef int ew = even_region[2]
    cdef int eh = even_region[3]

    # Handle empty regions
    if ow == 0 or oh == 0:
        if ew == 0 or eh == 0:
            return ((0, 0, img_width, img_height), (0, 0, img_width, img_height))
        return ((ex, ey, ew, eh), (ex, ey, ew, eh))
    if ew == 0 or eh == 0:
        return ((ox, oy, ow, oh), (ox, oy, ow, oh))

    # C# lines 1764-1767: Unify Y coordinates
    # int totalCropTop = Math.Min(oddCropRegion.Top, evenCropRegion.Top);
    # int totalCropBottom = Math.Max(oddCropRegion.Bottom, evenCropRegion.Bottom);
    # oddCropRegion = new Rectangle(oddCropRegion.X, totalCropTop, oddCropRegion.Width, totalCropBottom - totalCropTop);
    # evenCropRegion = new Rectangle(evenCropRegion.X, totalCropTop, evenCropRegion.Width, totalCropBottom - totalCropTop);
    cdef int totalCropTop = oy if oy < ey else ey  # Math.Min
    cdef int oddBottom = oy + oh
    cdef int evenBottom = ey + eh
    cdef int totalCropBottom = oddBottom if oddBottom > evenBottom else evenBottom  # Math.Max

    # Update Y and heights (width stays the same)
    oy = totalCropTop
    ey = totalCropTop
    oh = totalCropBottom - totalCropTop
    eh = totalCropBottom - totalCropTop

    # C# lines 1771-1772: Find max dimensions (after Y unification)
    cdef int maxWidth = ow if ow > ew else ew
    cdef int maxHeight = oh if oh > eh else eh

    # C# lines 1774-1775: Add margin (percentage of each dimension separately)
    # maxWidth += maxWidth * marginPercent / 100;
    # maxHeight += maxHeight * marginPercent / 100;
    cdef int marginW = maxWidth * margin_percent // 100
    cdef int marginH = maxHeight * margin_percent // 100
    maxWidth += marginW
    maxHeight += marginH

    cdef int dw, dh, newLeft, newTop

    # C# lines 1777-1798: Adjust odd region
    if ow < maxWidth or oh < maxHeight:
        dw = maxWidth - ow
        dh = maxHeight - oh
        newLeft = ox - dw // 2
        newTop = oy - dh // 2

        # C# lines 1786-1790: Clamp to boundaries
        if maxWidth > img_width:
            maxWidth = img_width
        if newLeft < 0:
            newLeft = 0
        elif newLeft > img_width - maxWidth:
            newLeft = img_width - maxWidth

        if maxHeight > img_height:
            maxHeight = img_height
        if newTop < 0:
            newTop = 0
        elif newTop > img_height - maxHeight:
            newTop = img_height - maxHeight

        ox = newLeft
        oy = newTop
        ow = maxWidth
        oh = maxHeight

    # C# lines 1800-1821: Adjust even region
    if ew < maxWidth or eh < maxHeight:
        dw = maxWidth - ew
        dh = maxHeight - eh
        newLeft = ex - dw // 2
        newTop = ey - dh // 2

        # C# lines 1808-1813: Clamp to boundaries
        if maxWidth > img_width:
            maxWidth = img_width
        if newLeft < 0:
            newLeft = 0
        elif newLeft > img_width - maxWidth:
            newLeft = img_width - maxWidth

        if maxHeight > img_height:
            maxHeight = img_height
        if newTop < 0:
            newTop = 0
        elif newTop > img_height - maxHeight:
            newTop = img_height - maxHeight

        ex = newLeft
        ey = newTop
        ew = maxWidth
        eh = maxHeight

    return ((ox, oy, ow, oh), (ex, ey, ew, eh))


# =============================================================================
# (F) IsPaperVerticalWriting_GetProbability
# C# lines 4693-4722
# =============================================================================

def IsPaperVerticalWriting_GetProbability(np.ndarray[uint8, ndim=2] image):
    """
    C# lines 4693-4722: public static double IsPaperVerticalWriting_GetProbability(Mat image)

    Detect if the page is primarily vertical (Japanese) or horizontal (Western) writing.
    Returns probability [0, 1] where >= 0.5 means vertical.

    Args:
        image: Grayscale image (CV_8UC1 / uint8 2D array)

    Returns:
        Probability of vertical writing [0.0, 1.0]
    """
    import cv2

    # C# line 4706: Horizontal score (scan rows for line patterns)
    cdef double horizontalScore = _ComputeLinearScore(image)

    # C# lines 4709-4711: Rotate 90° and compute vertical score
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cdef double verticalScore = _ComputeLinearScore(rotated)

    # C# lines 4714-4715: Normalize to probability
    cdef double total = horizontalScore + verticalScore + 1e-9
    cdef double verticalProbability = verticalScore / total

    # C# lines 4718-4719: Clamp to [0, 1]
    if verticalProbability < 0.0:
        verticalProbability = 0.0
    if verticalProbability > 1.0:
        verticalProbability = 1.0

    return verticalProbability


cdef double _ComputeLinearScore(np.ndarray[uint8, ndim=2] img):
    """
    C# lines 4727-4857: private static double ComputeLinearScore(Mat img)

    Scan image row by row, analyzing line patterns to detect text direction.
    """
    cdef int width = img.shape[1]
    cdef int height = img.shape[0]

    # C# lines 4732-4734: Divide into 4 blocks
    cdef int blockWidth = width // 4
    cdef double[4] blockScores
    cdef int blk, startX, endX
    cdef int y, x
    cdef int intersects
    cdef bint inBlack
    cdef long zeroLines, count
    cdef double mean, m2, delta, delta2
    cdef double variance, stddev, variationCoefficient, zeroRatio
    cdef double threshold
    cdef bint inLine, isLine
    cdef int runLen
    cdef double separationRatio, score
    cdef int medianLine, medianGap

    # Allocate intersections array
    cdef int* intersectionsPerRow = <int*>malloc(height * sizeof(int))
    if intersectionsPerRow == NULL:
        return 0.0

    # Lists for line/gap lengths
    cdef list lineThicknesses
    cdef list gapHeights

    for blk in range(4):
        startX = blk * blockWidth
        endX = width if blk == 3 else startX + blockWidth

        # C# lines 4742-4780: Count intersections per row
        zeroLines = 0
        mean = 0.0
        m2 = 0.0
        count = 0

        for y in range(height):
            intersects = 0
            inBlack = False

            # C# lines 4754-4768: Count black pixel clusters
            for x in range(startX, endX):
                if img[y, x] == 0:  # Black pixel
                    if not inBlack:
                        intersects += 1
                        inBlack = True
                else:
                    inBlack = False

            intersectionsPerRow[y] = intersects

            if intersects == 0:
                zeroLines += 1

            # C# lines 4775-4779: Welford's algorithm for mean/variance
            count += 1
            delta = intersects - mean
            mean += delta / count
            delta2 = intersects - mean
            m2 += delta * delta2

        # C# lines 4782-4786
        if count == 0:
            blockScores[blk] = 0.0
            continue

        # C# lines 4788-4791
        variance = m2 / count
        stddev = sqrt(variance)
        variationCoefficient = stddev / mean if mean > 0.0 else 0.0
        zeroRatio = <double>zeroLines / count

        # C# lines 4794-4829: Extract line/gap runs
        threshold = mean if mean > 1.0 else 1.0
        lineThicknesses = []
        gapHeights = []

        inLine = False
        runLen = 0

        for y in range(height):
            isLine = intersectionsPerRow[y] >= threshold

            if isLine == inLine:
                runLen += 1
            else:
                # Record previous run
                if runLen > 0:
                    if inLine:
                        lineThicknesses.append(runLen)
                    else:
                        gapHeights.append(runLen)
                inLine = isLine
                runLen = 1

        # Final run
        if runLen > 0:
            if inLine:
                lineThicknesses.append(runLen)
            else:
                gapHeights.append(runLen)

        # C# lines 4831-4840: Separation ratio
        separationRatio = 0.0
        if len(lineThicknesses) > 0 and len(gapHeights) > 0:
            medianLine = _Median2(lineThicknesses)
            medianGap = _Median2(gapHeights)
            separationRatio = <double>medianGap / (medianLine + medianGap + 1e-9)

        # C# lines 4844-4847: Weighted score
        score = (variationCoefficient * 0.4) + (zeroRatio * 0.2) + (separationRatio * 0.4)

        # C# lines 4849-4850: Clamp
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        blockScores[blk] = score

    free(intersectionsPerRow)

    # C# line 4856: Return average
    return (blockScores[0] + blockScores[1] + blockScores[2] + blockScores[3]) / 4.0


cdef int _Median2(list data):
    """
    C# lines 4862-4871: private static int Median2(List<int> data)
    """
    if data is None or len(data) == 0:
        return 0

    data_sorted = sorted(data)
    cdef int n = len(data_sorted)
    cdef int mid = n // 2

    if n % 2 == 1:
        return data_sorted[mid]
    else:
        return (data_sorted[mid - 1] + data_sorted[mid]) // 2


# =============================================================================
# (I) OcrGetWordBlocks - C# lines 4490-4654
# Extracts word/text block bounding boxes from a binarized page image
# =============================================================================

def OcrGetWordBlocks(np.ndarray[uint8, ndim=2] binary_image, tuple ignore_region=None):
    """
    C# lines 4490-4654: OcrGetWordBlocks

    Extracts text block bounding boxes from a binary image.
    Uses connected components analysis and clustering.

    Args:
        binary_image: Binary image (grayscale, 0/255)
        ignore_region: (x, y, width, height) tuple for region to ignore (center of page)

    Returns:
        List of (x, y, width, height) tuples for each text block
    """
    import cv2

    cdef int h = binary_image.shape[0]
    cdef int w = binary_image.shape[1]

    if h == 0 or w == 0:
        return []

    # C# lines 4499-4504: Make a working copy
    work = binary_image.copy()

    # C# lines 4502-4504: Ensure black text on white background
    # If more than 80% is white (255), invert
    cdef double white_ratio = <double>np.count_nonzero(work) / (h * w)
    if white_ratio > 0.8:
        work = cv2.bitwise_not(work)

    # C# lines 4506-4516: Black out the ignore region
    if ignore_region is not None:
        ix, iy, iw, ih = ignore_region
        ix = max(0, ix)
        iy = max(0, iy)
        iw = max(0, min(ix + iw, w) - ix)
        ih = max(0, min(iy + ih, h) - iy)
        if iw > 0 and ih > 0:
            work[iy:iy+ih, ix:ix+iw] = 0

    # C# lines 4518-4543: Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        work, connectivity=8, ltype=cv2.CV_32S
    )

    # Collect character rectangles
    char_rects = []
    char_heights = []
    char_widths = []

    cdef int label_id, comp_x, comp_y, comp_w, comp_h

    for label_id in range(1, num_labels):  # Skip background (0)
        comp_x = stats[label_id, cv2.CC_STAT_LEFT]
        comp_y = stats[label_id, cv2.CC_STAT_TOP]
        comp_w = stats[label_id, cv2.CC_STAT_WIDTH]
        comp_h = stats[label_id, cv2.CC_STAT_HEIGHT]

        # C# line 4535: Filter too small
        if comp_w < 3 or comp_h < 30:
            continue

        char_rects.append((comp_x, comp_y, comp_w, comp_h))
        char_widths.append(comp_w)
        char_heights.append(comp_h)

    if len(char_rects) == 0:
        return []

    # C# lines 4547-4548: Document median dimensions
    cdef int doc_med_h = _median_int(char_heights)
    cdef int doc_med_w = _median_int(char_widths)

    # C# lines 4550-4555: Remove horizontal lines
    def is_horizontal_line(rect):
        rx, ry, rw, rh = rect
        return rh <= doc_med_h * 0.7 and rw >= doc_med_w * 4

    char_rects = [r for r in char_rects if not is_horizontal_line(r)]
    if len(char_rects) == 0:
        return []

    # C# lines 4557-4586: Cluster nearby characters using Union-Find
    cdef int x_pad = max(3, int(doc_med_w * 0.6))
    cdef int y_pad = max(1, int(doc_med_h * 0.2))

    cdef int n = len(char_rects)

    # Create inflated rectangles
    inflated = []
    for rx, ry, rw, rh in char_rects:
        inflated.append((
            max(0, rx - x_pad),
            max(0, ry - y_pad),
            rw + x_pad * 2,
            rh + y_pad * 2
        ))

    # Union-Find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Check for overlaps
    cdef int i, j
    for i in range(n):
        for j in range(i + 1, n):
            # Check if inflated rectangles intersect
            ax, ay, aw, ah = inflated[i]
            bx, by, bw, bh = inflated[j]

            # Intersection check
            if ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by:
                union(i, j)

    # Group by cluster
    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(char_rects[i])

    # C# lines 4596-4653: Process each cluster
    words = []

    for group in clusters.values():
        if len(group) == 0:
            continue

        # Cluster statistics
        group_heights = [r[3] for r in group]
        group_widths = [r[2] for r in group]
        med_h = _median_int(group_heights)
        med_w = _median_int(group_widths)

        # C# lines 4606-4623: Filter out slim vertical lines
        mean_x = sum(r[0] + r[2] / 2.0 for r in group) / len(group)

        filtered = []
        for rx, ry, rw, rh in sorted(group, key=lambda r: r[0]):
            slim = rw <= max(5, med_w * 0.7)
            asp6 = rh / max(1, rw) >= 6.0
            left_of_center = rx + rw <= mean_x - med_w * 0.3

            if slim and asp6 and left_of_center:
                continue  # Skip vertical lines
            filtered.append((rx, ry, rw, rh))

        if len(filtered) == 0:
            continue

        # Recalculate median with filtered
        med_h = _median_int([r[3] for r in filtered])
        med_w = _median_int([r[2] for r in filtered])

        # C# lines 4630-4638: Filter vertical lines again
        def is_vertical_line(rect):
            rx, ry, rw, rh = rect
            narrow = rw <= max(5, med_w * 0.7)
            taller = rh >= med_h * 1.1
            asp_long = rh / max(1, rw) >= 6.0
            return narrow and taller and asp_long

        pure_chars = [r for r in filtered if not is_vertical_line(r)]
        if len(pure_chars) == 0:
            continue

        # C# lines 4642-4646: Compute union rectangle with margin
        min_x = min(r[0] for r in pure_chars)
        min_y = min(r[1] for r in pure_chars)
        max_x = max(r[0] + r[2] for r in pure_chars)
        max_y = max(r[1] + r[3] for r in pure_chars)

        margin = max(2, med_h // 10)
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x += margin
        max_y += margin

        # Clamp to image bounds
        if min_x >= 0 and min_y >= 0 and max_x <= w and max_y <= h:
            words.append((min_x, min_y, max_x - min_x, max_y - min_y))

    # C# lines 4648-4653: Sort by reading order (top to bottom, left to right)
    words.sort(key=lambda r: (r[1], r[0]))

    return words


cdef int _median_int(list data):
    """
    C# line 4399: Median(List<int> v) { v.Sort(); return v[v.Count / 2]; }

    C# uses upper median (index = count / 2), not statistical median.
    Example: [1,2,3,4] -> sorted -> v[4/2] = v[2] = 3
    """
    if data is None or len(data) == 0:
        return 0
    data_sorted = sorted(data)
    return data_sorted[len(data_sorted) // 2]


# =============================================================================
# (J-0) PnOcrProcessForBook - C# lines 3229-3620
# Sophisticated page number detection with cross-page validation
# =============================================================================

class PnOcrCandidate:
    """One possible page number candidate detected on a page."""
    __slots__ = ['text', 'text_int', 'bbox', 'possibility']

    def __init__(self, text, text_int, bbox, possibility=1.0):
        self.text = text
        self.text_int = text_int
        self.bbox = bbox  # (x, y, w, h)
        self.possibility = possibility


class PnOcrPageResult:
    """OCR results for a single page - all number candidates."""
    __slots__ = ['physical_page', 'candidates', 'candidates_by_int', 'logical_page',
                 'found_bbox', 'std_point']

    def __init__(self, physical_page):
        self.physical_page = physical_page
        self.candidates = []  # List of PnOcrCandidate
        self.candidates_by_int = {}  # Dict[int, List[PnOcrCandidate]]
        self.logical_page = None
        self.found_bbox = None  # The bbox of the matched page number
        self.std_point = None  # (x, y) reference point for shift calculation

    def add_candidate(self, text, text_int, bbox, possibility=1.0):
        """Add a candidate and index by integer value."""
        cand = PnOcrCandidate(text, text_int, bbox, possibility)
        self.candidates.append(cand)
        if text_int not in self.candidates_by_int:
            self.candidates_by_int[text_int] = []
        self.candidates_by_int[text_int].append(cand)


# =============================================================================
# Helper functions for PnOcrProcessForBook - C# faithful ports (Cython/C)
# =============================================================================

cdef int _levenshtein_distance(str s, str t):
    """
    C# Helper.cs lines 4014-4044: Levenshtein distance with O(min(n,m)) memory.
    """
    cdef int n = len(s)
    cdef int m = len(t)
    cdef int i, j, cost
    cdef int* v0
    cdef int* v1
    cdef int* tmp
    cdef int result

    if n == 0:
        return m
    if m == 0:
        return n

    # Allocate arrays
    v0 = <int*>malloc((m + 1) * sizeof(int))
    v1 = <int*>malloc((m + 1) * sizeof(int))

    # Initialize v0 = [0, 1, 2, ..., m]
    for j in range(m + 1):
        v0[j] = j

    for i in range(n):
        v1[0] = i + 1  # First column (insertion cost)

        for j in range(m):
            cost = 0 if s[i] == t[j] else 1

            # v1[j+1] = min(insertion, deletion, substitution)
            v1[j + 1] = v1[j] + 1  # insertion
            if v0[j + 1] + 1 < v1[j + 1]:
                v1[j + 1] = v0[j + 1] + 1  # deletion
            if v0[j] + cost < v1[j + 1]:
                v1[j + 1] = v0[j] + cost  # substitution

        # Swap rows
        tmp = v0
        v0 = v1
        v1 = tmp

    result = v0[m]
    free(v0)
    free(v1)
    return result


cpdef double _get_two_string_similarity(str s1, str s2):
    """
    C# Helper.cs lines 3996-4011: String similarity using Levenshtein distance.
    Returns 1.0 for identical strings, 0.0 for completely different.
    """
    cdef int distance, max_len
    cdef int len1 = len(s1) if s1 else 0
    cdef int len2 = len(s2) if s2 else 0

    # Both empty = perfect match
    if len1 == 0 and len2 == 0:
        return 1.0

    # One empty = no match
    if len1 == 0 or len2 == 0:
        return 0.0

    distance = _levenshtein_distance(s1, s2)
    max_len = len1 if len1 > len2 else len2

    # Higher distance = lower similarity
    return 1.0 - <double>distance / <double>max_len


cdef tuple _calc_most_overlap_rect_center(list rects):
    """
    C# SuperPdfUtil.cs lines 231-323: CalcMostOverlapRect
    Find the cell with maximum overlap count using sweep line algorithm.
    Returns (center_x, center_y) of the most overlapping region.

    Args:
        rects: List of (x, y, w, h) tuples

    Returns:
        (center_x, center_y) tuple
    """
    cdef int num_rects = len(rects)
    cdef int i, xi, yi
    cdef int x, y, w, h
    cdef int left, right, top, bottom
    cdef int r_left, r_right, r_top, r_bottom
    cdef int rx, ry, rw, rh
    cdef int count, best_count
    cdef double dist2, best_dist2
    cdef double cell_cx, cell_cy
    cdef double global_left, global_top, global_right, global_bottom
    cdef double global_center_x, global_center_y
    cdef int best_left, best_top, best_width, best_height
    cdef bint have_best = False

    if num_rects == 0:
        return (0, 0)

    # C# lines 245-258: Collect unique edges
    x_edge_set = set()
    y_edge_set = set()

    for i in range(num_rects):
        x, y, w, h = rects[i]
        x_edge_set.add(x)
        x_edge_set.add(x + w)  # right edge
        y_edge_set.add(y)
        y_edge_set.add(y + h)  # bottom edge

    x_edges = sorted(x_edge_set)
    y_edges = sorted(y_edge_set)

    cdef int num_x_edges = len(x_edges)
    cdef int num_y_edges = len(y_edges)

    # C# lines 260-267: Global bounding box center
    x, y, w, h = rects[0]
    global_left = <double>x
    global_top = <double>y
    global_right = <double>(x + w)
    global_bottom = <double>(y + h)

    for i in range(1, num_rects):
        x, y, w, h = rects[i]
        if x < global_left:
            global_left = x
        if y < global_top:
            global_top = y
        if x + w > global_right:
            global_right = x + w
        if y + h > global_bottom:
            global_bottom = y + h

    global_center_x = (global_left + global_right) / 2.0
    global_center_y = (global_top + global_bottom) / 2.0

    # C# lines 269-315: Cell scan
    best_count = -1
    best_dist2 = 1e18  # Large number

    for xi in range(num_x_edges - 1):
        left = x_edges[xi]
        right = x_edges[xi + 1]
        if left == right:
            continue  # Zero-width cell

        for yi in range(num_y_edges - 1):
            top = y_edges[yi]
            bottom = y_edges[yi + 1]
            if top == bottom:
                continue  # Zero-height cell

            # Count rectangles that fully contain this cell
            count = 0
            for i in range(num_rects):
                rx, ry, rw, rh = rects[i]
                r_left = rx
                r_right = rx + rw
                r_top = ry
                r_bottom = ry + rh
                if r_left <= left and r_right >= right and r_top <= top and r_bottom >= bottom:
                    count += 1

            if count == 0:
                continue

            # Update condition: more overlap, or same overlap but closer to center
            cell_cx = (left + right) / 2.0
            cell_cy = (top + bottom) / 2.0
            dist2 = (cell_cx - global_center_x) * (cell_cx - global_center_x) + \
                    (cell_cy - global_center_y) * (cell_cy - global_center_y)

            if count > best_count or (count == best_count and dist2 < best_dist2):
                best_count = count
                best_dist2 = dist2
                best_left = left
                best_top = top
                best_width = right - left
                best_height = bottom - top
                have_best = True

    if not have_best or best_count <= 0:
        # Fallback: use global center
        return (<int>global_center_x, <int>global_center_y)

    # Return center of best rect
    return (best_left + best_width // 2, best_top + best_height // 2)


def PnOcrProcessForBook(list page_ocr_results, int image_width, int image_height):
    """
    C# lines 3229-3620: Process OCR results for entire book to find correct page numbers.

    This implements the sophisticated cross-page validation algorithm:
    1. Try shifts -300 to +300 to find correct physical→logical mapping
    2. Find standard bounding box where page numbers appear
    3. Match each page's correct page number within standard bbox
    4. Calculate alignment shifts

    Args:
        page_ocr_results: List of PnOcrPageResult objects (one per page)
        image_width: Width of page images
        image_height: Height of page images

    Returns:
        List of (shift_x, shift_y, logical_page) tuples for each page
    """
    cdef int n = len(page_ocr_results)
    cdef int i
    cdef int center_x
    cdef int group_num
    cdef int page_num
    cdef int shift_test
    cdef int kari_page
    cdef double possibility_sum
    cdef int best_shift = 0
    cdef double best_score = 0.0
    cdef int num_matched
    cdef int max_physical
    cdef int margin_width
    cdef int margin_height
    cdef bint is_group0_right = False

    if n == 0:
        return []

    # C# lines 3335-3356: Group pages by odd/even physical page number
    cdef list group0 = []  # Even physical pages (0, 2, 4, ...)
    cdef list group1 = []  # Odd physical pages (1, 3, 5, ...)

    for i in range(n):
        page = page_ocr_results[i]
        if page.physical_page % 2 == 0:
            group0.append(page)
        else:
            group1.append(page)

    groups = [group0, group1]

    # C# lines 3358-3390: Find the correct shift between physical and logical page numbers
    # Try shifts from -300 to +300, find the one with highest match score
    shift_scores = {}
    shift_match_counts = {}

    for shift_test in range(-300, 300):
        possibility_sum = 0.0
        match_count = 0

        for group in groups:
            for page in group:
                kari_page = page.physical_page - shift_test
                if kari_page >= 1:
                    if kari_page in page.candidates_by_int:
                        for cand in page.candidates_by_int[kari_page]:
                            possibility_sum += cand.possibility
                        match_count += 1

        if possibility_sum > 0:
            shift_scores[shift_test] = possibility_sum
            shift_match_counts[shift_test] = match_count

    # Find best shift (highest score, prefer smaller absolute shift for ties)
    for shift_test, score in sorted(shift_scores.items(), key=lambda x: (-x[1], abs(x[0]))):
        best_shift = shift_test
        best_score = score
        break

    num_matched = shift_match_counts.get(best_shift, 0)
    max_physical = max(p.physical_page for p in page_ocr_results) if page_ocr_results else 0

    # C# lines 3392-3396: Validate - need at least 5 matches or 1/3 of pages
    if num_matched < 5 or (num_matched * 3) < max_physical:
        # Not enough matches - return zero shifts
        return [(0, 0, None) for _ in range(n)]

    # C# lines 3398-3400: Margin sizes for bounding box matching
    # C# uses constant internalHighResImgHeight (7016), not passed image_height
    margin_width = int(INTERNAL_HIGH_RES_HEIGHT * 0.030)   # = 210
    margin_height = int(INTERNAL_HIGH_RES_HEIGHT * 0.025)  # = 175

    # C# lines 3406-3473: Find standard bounding box for each odd/even group
    # This is where page numbers are expected to appear
    group_std_bboxes = [None, None]

    for group_num in range(2):
        group = groups[group_num]
        if not group:
            continue

        # Collect bboxes of all correct page number matches
        bbox_candidates = []
        for page in group:
            page_num = page.physical_page - best_shift
            if page_num >= 1 and page_num in page.candidates_by_int:
                for cand in page.candidates_by_int[page_num]:
                    # Add margin to bbox
                    bx, by, bw, bh = cand.bbox
                    expanded_bbox = (
                        bx - margin_width,
                        by - margin_height,
                        bw + margin_width * 2,
                        bh + margin_height * 2
                    )
                    bbox_candidates.append({
                        'bbox_margin': expanded_bbox,
                        'bbox_orig': cand.bbox,
                        'matches': 0
                    })

        if not bbox_candidates:
            continue

        # C# lines 3429-3450: Count how many pages each candidate bbox covers
        for candidate in bbox_candidates:
            cbx, cby, cbw, cbh = candidate['bbox_margin']
            for page in group:
                page_num = page.physical_page - best_shift
                if page_num >= 1 and page_num in page.candidates_by_int:
                    for cand in page.candidates_by_int[page_num]:
                        # Check if cand.bbox is inside candidate['bbox_margin']
                        px, py, pw, ph = cand.bbox
                        if (px >= cbx and py >= cby and
                            px + pw <= cbx + cbw and py + ph <= cby + cbh):
                            candidate['matches'] += 1

        # C# lines 3454-3466: Select best bounding box region
        if bbox_candidates:
            max_matches = max(c['matches'] for c in bbox_candidates)
            threshold = int(max_matches * 0.7)

            # Filter to candidates with ≥70% of max matches
            good_candidates = [c for c in bbox_candidates if c['matches'] >= threshold]

            if good_candidates:
                # C# line 3457: Sort by area (smallest first), take top 30%
                good_candidates.sort(key=lambda c: c['bbox_margin'][2] * c['bbox_margin'][3])
                take_count = max(1, int(math.ceil(0.3 * len(good_candidates))))
                top_30_pct = good_candidates[:take_count]

                # C# line 3460: Find most overlapping rect center using sweep line algorithm
                rects = [c['bbox_margin'] for c in top_30_pct]
                center_x, center_y = _calc_most_overlap_rect_center(rects)

                # C# lines 3463-3464: Get median width/height from ALL ≥70% match candidates
                widths = sorted([c['bbox_margin'][2] for c in good_candidates])
                heights = sorted([c['bbox_margin'][3] for c in good_candidates])
                # ElementAtPosition(0.5) uses MidpointRounding.AwayFromZero
                # Emulate with floor(x + 0.5) instead of Python's banker's rounding
                med_width = widths[int(math.floor(0.5 * (len(widths) - 1) + 0.5))]
                med_height = heights[int(math.floor(0.5 * (len(heights) - 1) + 0.5))]

                group_std_bboxes[group_num] = (
                    center_x - med_width // 2,
                    center_y - med_height // 2,
                    med_width,
                    med_height
                )

    # C# lines 3475-3483: Determine which group is right-side
    if group_std_bboxes[0] is not None and group_std_bboxes[1] is not None:
        # Compare left edges
        if group_std_bboxes[0][0] > group_std_bboxes[1][0] + group_std_bboxes[1][2]:
            is_group0_right = True

    # C# lines 3506-3564: For each page, find the best matching page number bbox
    center_x = image_width // 2

    for group_num in range(2):
        group = groups[group_num]
        std_bbox = group_std_bboxes[group_num]
        is_right_side = (group_num == 0) if is_group0_right else (group_num == 1)

        for page in group:
            page_num = page.physical_page - best_shift
            page.logical_page = page_num if page_num >= 1 else None

            if page_num < 1:
                continue

            found_cand = None

            if std_bbox is not None:
                sbx, sby, sbw, sbh = std_bbox
                # C# line 3492: Calculate center point for distance sorting
                basic_center_x = sbx + sbw // 2
                basic_center_y = sby + sbh // 2

                def is_inside_std_bbox(cand):
                    px, py, pw, ph = cand.bbox
                    return (px >= sbx and py >= sby and
                            px + pw <= sbx + sbw and py + ph <= sby + sbh)

                def distance_to_center(cand):
                    px, py, pw, ph = cand.bbox
                    cx = px + pw // 2
                    cy = py + ph // 2
                    return (cx - basic_center_x) ** 2 + (cy - basic_center_y) ** 2

                # C# lines 3518-3522: Try exact page number match within std_bbox
                if page_num in page.candidates_by_int:
                    matches_in_bbox = [c for c in page.candidates_by_int[page_num] if is_inside_std_bbox(c)]
                    if matches_in_bbox:
                        # Sort by distance to center, then by area (smaller is better)
                        matches_in_bbox.sort(key=lambda c: (distance_to_center(c), c.bbox[2] * c.bbox[3]))
                        found_cand = matches_in_bbox[0]

                # C# lines 3524-3529: Fallback 1 - similar text within std_bbox
                if found_cand is None:
                    page_num_str = str(page_num)
                    candidates_in_bbox = [c for c in page.candidates if c.text and is_inside_std_bbox(c)]
                    if candidates_in_bbox:
                        # Sort by text similarity (descending), then distance, then area
                        # C# uses Levenshtein-based similarity
                        candidates_in_bbox.sort(key=lambda c: (
                            -_get_two_string_similarity(c.text, page_num_str),
                            distance_to_center(c),
                            c.bbox[2] * c.bbox[3]
                        ))
                        found_cand = candidates_in_bbox[0]

                # C# lines 3532-3536: Fallback 2 - any OCR text within std_bbox
                if found_cand is None:
                    candidates_in_bbox = [c for c in page.candidates if c.text and is_inside_std_bbox(c)]
                    if candidates_in_bbox:
                        candidates_in_bbox.sort(key=lambda c: (distance_to_center(c), c.bbox[2] * c.bbox[3]))
                        found_cand = candidates_in_bbox[0]

                # C# lines 3538-3542: Fallback 3 - any entry within std_bbox
                if found_cand is None:
                    candidates_in_bbox = [c for c in page.candidates if is_inside_std_bbox(c)]
                    if candidates_in_bbox:
                        candidates_in_bbox.sort(key=lambda c: (distance_to_center(c), c.bbox[2] * c.bbox[3]))
                        found_cand = candidates_in_bbox[0]

            # If still not found, leave found_cand as None (no shift for this page)

            if found_cand is not None:
                page.found_bbox = found_cand.bbox
                bx, by, bw, bh = found_cand.bbox

                # C# lines 3553-3560: Calculate std_point (reference for shift)
                if is_right_side:
                    std_x = bx + bw  # Right edge
                else:
                    std_x = bx  # Left edge
                std_y = by + bh // 2  # Center Y

                page.std_point = (std_x, std_y)

    # C# lines 3567-3617: Calculate average positions and shifts
    # First, align Y positions between odd/even groups
    cdef list odd_ys = []
    cdef list even_ys = []

    for page in group1:  # Odd physical pages
        if page.std_point is not None:
            odd_ys.append(page.std_point[1])
    for page in group0:  # Even physical pages
        if page.std_point is not None:
            even_ys.append(page.std_point[1])

    # C# truncates average FIRST, then computes difference (integer arithmetic)
    cdef int avg_y_odd = <int>(sum(odd_ys) / len(odd_ys)) if odd_ys else 0
    cdef int avg_y_even = <int>(sum(even_ys) / len(even_ys)) if even_ys else 0
    cdef int zure = abs(avg_y_odd - avg_y_even)

    cdef int additional_shift_y_odd = 0
    cdef int additional_shift_y_even = 0
    cdef int avg_y_both = 0

    if zure < <int>(margin_height * 2.0) and odd_ys and even_ys:
        avg_y_both = (avg_y_odd + avg_y_even) // 2
        additional_shift_y_odd = avg_y_both - avg_y_odd
        additional_shift_y_even = avg_y_both - avg_y_even

    # Calculate shifts for each group
    results = [(0, 0, None) for _ in range(n)]
    cdef int avg_x = 0
    cdef int avg_y = 0
    cdef int shift_x = 0
    cdef int shift_y = 0

    for group_num in range(2):
        group = groups[group_num]
        additional_shift_y = additional_shift_y_odd if group_num == 1 else additional_shift_y_even

        # Calculate average X for this group (C# truncates average first)
        xs = [p.std_point[0] for p in group if p.std_point is not None]
        if not xs:
            continue
        avg_x = <int>(sum(xs) / len(xs))
        avg_y = avg_y_odd if group_num == 1 else avg_y_even

        for page in group:
            if page.std_point is not None:
                shift_x = avg_x - page.std_point[0]  # Integer subtraction
                shift_y = avg_y - page.std_point[1] + additional_shift_y  # Integer subtraction

                # Find index in original list
                for idx in range(n):
                    if page_ocr_results[idx].physical_page == page.physical_page:
                        results[idx] = (shift_x, shift_y, page.logical_page)
                        break

    return results


# =============================================================================
# (J) CalculatePageAlignmentShifts - C# lines 3617-3667
# Calculates shift_x/shift_y for page alignment based on page number positions
# =============================================================================

def CalculatePageAlignmentShifts(list page_number_bboxes, list is_odd_flags, int image_width):
    """
    C# lines 3617-3667: Calculate page alignment shifts based on page number positions.

    For each page with a detected page number, calculates how much to shift
    the page so all page numbers align to the same position.

    Args:
        page_number_bboxes: List of (x, y, w, h) tuples for detected page numbers (None if not detected)
        is_odd_flags: List of booleans indicating if each page is odd
        image_width: Width of the page images

    Returns:
        List of (shift_x, shift_y) tuples for each page
    """
    cdef int n = len(page_number_bboxes)
    if n == 0:
        return []

    # Initialize all shifts to 0
    shifts = [(0, 0) for _ in range(n)]

    # Separate odd and even pages
    odd_pages = []  # List of (index, bbox, std_point)
    even_pages = []

    cdef int i
    cdef int center_x = image_width // 2

    for i in range(n):
        bbox = page_number_bboxes[i]
        if bbox is None:
            continue

        bx, by, bw, bh = bbox
        is_odd = is_odd_flags[i]

        # C# lines 3600-3612: Determine reference point
        # Right side: use right edge, Left side: use left edge
        is_right_side = (bx + bw // 2) >= center_x

        if is_right_side:
            std_x = bx + bw  # Right edge
        else:
            std_x = bx  # Left edge

        std_y = by + bh // 2  # Center Y

        if is_odd:
            odd_pages.append((i, (std_x, std_y)))
        else:
            even_pages.append((i, (std_x, std_y)))

    # C# lines 3617-3637: Calculate average Y per group and align if close
    cdef double avg_y_odd = 0.0
    cdef double avg_y_even = 0.0
    cdef int additional_shift_y_odd = 0
    cdef int additional_shift_y_even = 0

    if len(odd_pages) > 0:
        avg_y_odd = sum(p[1][1] for p in odd_pages) / len(odd_pages)
    if len(even_pages) > 0:
        avg_y_even = sum(p[1][1] for p in even_pages) / len(even_pages)

    cdef double zure = abs(avg_y_odd - avg_y_even)
    cdef int margin_height = image_width // 20  # Approximate margin

    if zure < margin_height * 2.0 and len(odd_pages) > 0 and len(even_pages) > 0:
        # Align odd/even Y positions
        # C#: Truncate avg_y_both FIRST, then subtract (lines 3593-3606)
        avg_y_both_int = int((avg_y_odd + avg_y_even) / 2.0)
        avg_y_odd_int = int(avg_y_odd)
        avg_y_even_int = int(avg_y_even)
        additional_shift_y_odd = avg_y_both_int - avg_y_odd_int
        additional_shift_y_even = avg_y_both_int - avg_y_even_int

    # C# lines 3639-3667: Calculate shifts for each page
    # C#: int averageX = (int)Average(); shift_x = averageX - page_x;
    # Truncate average FIRST, then subtract
    # Process odd pages
    if len(odd_pages) > 0:
        avg_x_odd_int = int(sum(p[1][0] for p in odd_pages) / len(odd_pages))
        avg_y_odd_int = int(avg_y_odd)

        for idx, (page_x, page_y) in odd_pages:
            shift_x = avg_x_odd_int - page_x
            shift_y = avg_y_odd_int - page_y + additional_shift_y_odd
            shifts[idx] = (shift_x, shift_y)

    # Process even pages
    if len(even_pages) > 0:
        avg_x_even_int = int(sum(p[1][0] for p in even_pages) / len(even_pages))
        avg_y_even_int = int(avg_y_even)

        for idx, (page_x, page_y) in even_pages:
            shift_x = avg_x_even_int - page_x
            shift_y = avg_y_even_int - page_y + additional_shift_y_even
            shifts[idx] = (shift_x, shift_y)

    return shifts


# =============================================================================
# Export constants
# =============================================================================

def get_internal_high_res_width():
    return INTERNAL_HIGH_RES_WIDTH

def get_internal_high_res_height():
    return INTERNAL_HIGH_RES_HEIGHT

def get_final_target_height():
    return FINAL_TARGET_HEIGHT
