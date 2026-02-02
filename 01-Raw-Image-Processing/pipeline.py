"""
RAW Image Processing Pipeline
"""

from typing import Tuple
import numpy as np
import scipy.ndimage
from PIL import Image
import PIL.TiffTags


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def read_raw_metadata(filename: str) -> Tuple[int, Tuple[float, float, float]]:
    """
    Extract metadata from a RAW image file.
    
    Reads the black level and neutral white point values from the
    TIFF metadata embedded in RAW image files.
    
    Args:
        filename: Path to the RAW image file (DNG, TIFF, etc.)
    
    Returns:
        Tuple containing:
            - black_level: The sensor's black level value (integer)
            - as_shot_neutral: RGB tuple representing the neutral white point
    
    Example:
        >>> black_level, neutral = read_raw_metadata("image.dng")
        >>> print(f"Black level: {black_level}")
    """
    img = Image.open(filename)
    
    # Build metadata dictionary with human-readable tag names
    meta_dict = {
        PIL.TiffTags.TAGS.get(tag_id, tag_id): value 
        for tag_id, value in img.tag_v2.items()
    }
    
    black_level = int(meta_dict.get('BlackLevel', 0))
    as_shot_neutral = tuple(meta_dict.get('AsShotNeutral', (1.0, 1.0, 1.0)))
    
    return black_level, as_shot_neutral


# =============================================================================
# BLACK LEVEL CORRECTION
# =============================================================================

def normalize_black_level(image: np.ndarray, black_level: float, 
                          white_level: float = 65535.0) -> np.ndarray:
    """
    Normalize image values from sensor range to [0, 1].
    
    Adjusts contrast so that the black level maps to 0 and the
    white level maps to 1. Values below black level are clipped to 0.
    
    Args:
        image: Input RAW image array
        black_level: Sensor's black level value
        white_level: Maximum sensor value (default: 65535 for 16-bit)
    
    Returns:
        Normalized image with values in range [0, 1]
    """
    # Linear transformation from [black_level, white_level] to [0, 1]
    result = (image.astype(np.float64) - black_level) / (white_level - black_level)
    
    # Clip to ensure values stay in valid range
    return np.clip(result, 0, 1)


# =============================================================================
# DEMOSAICING (BAYER PATTERN INTERPOLATION)
# =============================================================================

def extract_bayer_channels(image: np.ndarray, 
                           pattern: str = 'RGGB') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract RGB channels from a Bayer-pattern sensor image.
    
    Separates the mosaiced sensor data into individual color channels
    based on the Bayer pattern arrangement.
    
    Args:
        image: Bayer-pattern image from camera sensor
        pattern: Bayer pattern type ('RGGB', 'BGGR', 'GRBG', 'GBRG')
    
    Returns:
        Tuple of (R, G, B) arrays with extracted pixel values
        (non-sampled positions contain zeros)
    """
    R = np.zeros(image.shape)
    G = np.zeros(image.shape)
    B = np.zeros(image.shape)
    
    # Extract channels based on pattern layout
    if pattern == 'RGGB':
        R[0::2, 0::2] = image[0::2, 0::2]  # Red: top-left of 2x2 blocks
        G[0::2, 1::2] = image[0::2, 1::2]  # Green: top-right
        G[1::2, 0::2] = image[1::2, 0::2]  # Green: bottom-left
        B[1::2, 1::2] = image[1::2, 1::2]  # Blue: bottom-right
    elif pattern == 'BGGR':
        B[0::2, 0::2] = image[0::2, 0::2]
        G[0::2, 1::2] = image[0::2, 1::2]
        G[1::2, 0::2] = image[1::2, 0::2]
        R[1::2, 1::2] = image[1::2, 1::2]
    elif pattern == 'GRBG':
        G[0::2, 0::2] = image[0::2, 0::2]
        R[0::2, 1::2] = image[0::2, 1::2]
        B[1::2, 0::2] = image[1::2, 0::2]
        G[1::2, 1::2] = image[1::2, 1::2]
    elif pattern == 'GBRG':
        G[0::2, 0::2] = image[0::2, 0::2]
        B[0::2, 1::2] = image[0::2, 1::2]
        R[1::2, 0::2] = image[1::2, 0::2]
        G[1::2, 1::2] = image[1::2, 1::2]
    else:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")
    
    return R, G, B


def apply_neutral_white(R: np.ndarray, G: np.ndarray, B: np.ndarray,
                        neutral: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply neutral white point correction to color channels.
    
    Divides each channel by its corresponding neutral value to
    correct for the camera's color sensitivity.
    
    Args:
        R, G, B: Color channel arrays
        neutral: RGB neutral white point values
    
    Returns:
        Corrected (R, G, B) channel arrays
    """
    return R / neutral[0], G / neutral[1], B / neutral[2]


def interpolate_channels(R: np.ndarray, G: np.ndarray, 
                         B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate missing color values using bilinear filtering.
    
    Fills in the missing color values at each pixel position using
    weighted averages of neighboring pixels.
    
    Args:
        R, G, B: Sparse color channels from Bayer extraction
    
    Returns:
        Fully interpolated (R, G, B) channel arrays
    """
    # Green uses a cross-shaped kernel (4 neighbors)
    green_kernel = np.array([
        [0.0,  0.25, 0.0 ],
        [0.25, 1.0,  0.25],
        [0.0,  0.25, 0.0 ]
    ])
    
    # Red/Blue use a square kernel (8 neighbors + center)
    rb_kernel = np.array([
        [0.25, 0.5, 0.25],
        [0.5,  1.0, 0.5 ],
        [0.25, 0.5, 0.25]
    ])
    
    R_interp = scipy.ndimage.correlate(R, rb_kernel, mode='constant')
    G_interp = scipy.ndimage.correlate(G, green_kernel, mode='constant')
    B_interp = scipy.ndimage.correlate(B, rb_kernel, mode='constant')
    
    return R_interp, G_interp, B_interp


def merge_channels(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Combine separate RGB channels into a single color image.
    
    Args:
        R, G, B: Individual color channel arrays
    
    Returns:
        Combined RGB image array with shape (height, width, 3)
    """
    return np.stack((R, G, B), axis=2)


def demosaic(image: np.ndarray, pattern: str = 'RGGB',
             neutral: Tuple[float, float, float] = None) -> np.ndarray:
    """
    Complete demosaicing pipeline for Bayer-pattern images.
    
    Performs the full demosaicing process: channel extraction,
    optional neutral white correction, interpolation, and merging.
    
    Args:
        image: Bayer-pattern sensor image
        pattern: Bayer pattern type
        neutral: Optional neutral white point for correction
    
    Returns:
        Full-color RGB image
    """
    R, G, B = extract_bayer_channels(image, pattern)
    
    if neutral is not None:
        R, G, B = apply_neutral_white(R, G, B, neutral)
    
    R, G, B = interpolate_channels(R, G, B)
    
    return merge_channels(R, G, B)


# =============================================================================
# WHITE BALANCE
# =============================================================================

def white_balance(image: np.ndarray, white_point: np.ndarray) -> np.ndarray:
    """
    Apply manual white balance correction.
    
    Adjusts the image so that the specified color becomes pure white.
    
    Args:
        image: RGB image array
        white_point: RGB color that should become white
    
    Returns:
        White-balanced image
    """
    # Prevent division by zero
    safe_white = np.where(white_point < 1e-10, 1e-10, white_point)
    
    result = image / safe_white
    
    # Clamp values to valid range
    return np.minimum(result, 1.0)


# =============================================================================
# GAMMA CORRECTION
# =============================================================================

def _rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using luminance weights."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.clip(gray, 0, 1)


def compute_brightness(image: np.ndarray) -> np.ndarray:
    """
    Calculate the brightness (luminance) of an image.
    
    Normalizes the image, converts to grayscale, then scales back.
    
    Args:
        image: RGB image array
    
    Returns:
        2D brightness array
    """
    max_value = np.max(image)
    if max_value == 0:
        return np.zeros(image.shape[:2])
    
    normalized = image / max_value
    grayscale = _rgb_to_grayscale(normalized)
    
    return grayscale * max_value


def compute_chromaticity(image: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    """
    Calculate the chromaticity (color ratios) of an image.
    
    Divides each color channel by the brightness to separate
    color from intensity information.
    
    Args:
        image: RGB image array
        brightness: 2D brightness array
    
    Returns:
        Chromaticity image (RGB ratios)
    """
    # Avoid division by zero
    safe_brightness = np.where(brightness < 1e-10, 1e-10, brightness)
    
    R = image[:, :, 0] / safe_brightness
    G = image[:, :, 1] / safe_brightness
    B = image[:, :, 2] / safe_brightness
    
    return np.dstack((R, G, B))


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to an image.
    
    Raises pixel values to the power of (1/gamma) to adjust
    the brightness response curve.
    
    Args:
        image: Input image array
        gamma: Gamma value (typically 2.2 for sRGB)
    
    Returns:
        Gamma-corrected image
    """
    # Ensure gamma is never zero
    safe_gamma = max(gamma, 1e-10)
    
    return np.power(image, 1.0 / safe_gamma)


def reconstruct_from_chromaticity(brightness_corrected: np.ndarray, 
                                   chromaticity: np.ndarray) -> np.ndarray:
    """
    Reconstruct a color image from brightness and chromaticity.
    
    Multiplies the corrected brightness with chromaticity to
    produce the final color image.
    
    Args:
        brightness_corrected: Gamma-corrected brightness (expanded to 3D)
        chromaticity: Color ratio image
    
    Returns:
        Reconstructed RGB image
    """
    R = brightness_corrected[:, :, 0] * chromaticity[:, :, 0]
    G = brightness_corrected[:, :, 1] * chromaticity[:, :, 1]
    B = brightness_corrected[:, :, 2] * chromaticity[:, :, 2]
    
    return np.dstack((R, G, B))


def apply_gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Apply gamma correction while preserving color relationships.
    
    Separates brightness from chromaticity, applies gamma to
    brightness only, then reconstructs the final image.
    
    Args:
        image: RGB image array
        gamma: Gamma value (default: 2.2 for sRGB)
    
    Returns:
        Gamma-corrected RGB image
    """
    brightness = compute_brightness(image)
    chromaticity = compute_chromaticity(image, brightness)
    
    brightness_corrected = gamma_correct(brightness, gamma)
    # Expand brightness to 3D for reconstruction
    brightness_3d = np.dstack([brightness_corrected] * 3)
    
    return reconstruct_from_chromaticity(brightness_3d, chromaticity)


# =============================================================================
# HISTOGRAM NORMALIZATION
# =============================================================================

def prepare_histogram_bounds(image: np.ndarray, low: float, 
                              high: float) -> Tuple[float, float]:
    """
    Calculate safe bounds for histogram normalization.
    
    Ensures low >= 0 and high <= max(image).
    
    Args:
        image: Input image array
        low: Desired black point
        high: Desired white point
    
    Returns:
        Tuple of (new_low, new_high) values
    """
    max_value = np.max(image)
    
    new_low = max(low, 0)
    new_high = min(high, max_value)
    
    return new_low, new_high


def normalize_histogram(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Normalize image histogram to map [low, high] to [0, 1].
    
    Args:
        image: Input image array
        low: Black point value
        high: White point value
    
    Returns:
        Normalized image (may contain values outside [0, 1])
    """
    return (image - low) / (high - low)


def clip_histogram(image: np.ndarray) -> np.ndarray:
    """
    Clip image values to the valid [0, 1] range.
    
    Args:
        image: Input image array
    
    Returns:
        Clipped image with all values in [0, 1]
    """
    result = np.where(image < 0, 0, image)
    result = np.where(result > 1, 1, result)
    return result


def histogram_stretch(image: np.ndarray, low: float = 0.0, 
                      high: float = None) -> np.ndarray:
    """
    Apply full histogram stretching pipeline.
    
    Args:
        image: Input image array
        low: Black point (default: 0)
        high: White point (default: max of image)
    
    Returns:
        Histogram-stretched image with values in [0, 1]
    """
    if high is None:
        high = np.max(image)
    
    new_low, new_high = prepare_histogram_bounds(image, low, high)
    result = normalize_histogram(image, new_low, new_high)
    return clip_histogram(result)


# =============================================================================
# BINARY THRESHOLDING
# =============================================================================

def threshold_binary(image: np.ndarray, threshold: float, 
                     invert: bool = False) -> np.ndarray:
    """
    Convert image to binary using a threshold.
    
    Values greater than the threshold become 1, others become 0.
    
    Args:
        image: Input image array
        threshold: Threshold value
        invert: If True, swap 0s and 1s in the result
    
    Returns:
        Binary image (float type with values 0.0 or 1.0)
    """
    result = (image > threshold).astype(float)
    
    if invert:
        result = 1.0 - result
    
    return result


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def process_raw_image(filename: str, 
                      bayer_pattern: str = 'RGGB',
                      gamma: float = 2.2,
                      manual_white_balance: np.ndarray = None) -> np.ndarray:
    """
    Process a RAW image through the complete ISP pipeline.
    
    This function chains all processing steps to convert a RAW
    sensor image into a viewable RGB image.
    
    Args:
        filename: Path to RAW image file
        bayer_pattern: Bayer pattern of the sensor
        gamma: Gamma correction value
        manual_white_balance: Optional manual white point for white balance
    
    Returns:
        Processed RGB image as float array with values in [0, 1]
    
    Example:
        >>> result = process_raw_image("photo.dng", gamma=2.2)
        >>> plt.imshow(result)
    """
    # Load image and metadata
    img = Image.open(filename)
    raw_image = np.array(img)
    black_level, neutral = read_raw_metadata(filename)
    
    # Step 1: Black level correction
    normalized = normalize_black_level(raw_image, black_level)
    
    # Step 2: Demosaicing
    rgb = demosaic(normalized, pattern=bayer_pattern, neutral=neutral)
    
    # Step 3: Optional manual white balance
    if manual_white_balance is not None:
        rgb = white_balance(rgb, manual_white_balance)
    
    # Step 4: Histogram normalization
    rgb = histogram_stretch(rgb)
    
    # Step 5: Gamma correction
    rgb = apply_gamma_correction(rgb, gamma)
    
    # Final clipping to ensure valid range
    return np.clip(rgb, 0, 1)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_image(image: np.ndarray, filename: str) -> None:
    """
    Save a processed image to file.
    
    Args:
        image: Image array with values in [0, 1]
        filename: Output filename (supports PNG, JPEG, TIFF, etc.)
    """
    # Convert to 8-bit and save
    img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_8bit).save(filename)


def load_image(filename: str) -> np.ndarray:
    """
    Load an image file as a numpy array.
    
    Args:
        filename: Path to image file
    
    Returns:
        Image as numpy array
    """
    return np.array(Image.open(filename))
