# RAW Image Processing Pipeline

A complete Python pipeline for processing RAW camera images (DNG, TIFF) into high-quality RGB images.

## Features

This pipeline implements the full ISP (Image Signal Processing) workflow:

- **Black Level Correction** - Normalize sensor values and remove dark current
- **Demosaicing** - Convert Bayer-pattern sensor data to full-color images
- **White Balance** - Correct color temperature automatically or manually
- **Gamma Correction** - Apply perceptual brightness curve (sRGB)
- **Histogram Normalization** - Stretch contrast for optimal dynamic range
- **Binary Thresholding** - Create masks and binary images

## Example Output

![Processed RAW Image](results/IMG_0_processed.png)

## Installation

```bash
pip install numpy scipy pillow
```

## Quick Start

```python
from pipeline import process_raw_image, save_image

# Process a RAW image with default settings
result = process_raw_image("photo.dng")

# Save the result
save_image(result, "photo_processed.png")
```

## Advanced Usage

### Step-by-Step Processing

```python
from pipeline import (
    read_raw_metadata,
    normalize_black_level,
    demosaic,
    white_balance,
    apply_gamma_correction,
    histogram_stretch,
    save_image
)
import numpy as np
from PIL import Image

# Load RAW image
img = Image.open("photo.dng")
raw = np.array(img)
black_level, neutral = read_raw_metadata("photo.dng")

# Process step by step
normalized = normalize_black_level(raw, black_level)
rgb = demosaic(normalized, pattern='RGGB', neutral=neutral)
rgb = histogram_stretch(rgb)
rgb = apply_gamma_correction(rgb, gamma=2.2)

save_image(rgb, "result.png")
```

### Manual White Balance

```python
from pipeline import process_raw_image, white_balance
import numpy as np

# Define a white point (sampled from a white area in the image)
white_point = np.array([0.9, 0.85, 0.95])

# Apply during processing
result = process_raw_image("photo.dng", manual_white_balance=white_point)
```

### Create Binary Mask

```python
from pipeline import threshold_binary, load_image

image = load_image("photo.png") / 255.0
mask = threshold_binary(image, threshold=0.5, invert=False)
```

## Supported Bayer Patterns

- `RGGB` - Most common (Canon, Nikon, Sony)
- `BGGR` - Some sensors
- `GRBG` - Some sensors
- `GBRG` - Some sensors

## Limitations

This pipeline uses **bilinear interpolation** for demosaicing, which is fast but can produce visible artifacts:

- **Color fringing** at high-contrast edges (red/blue halos)
- **Zipper artifacts** on fine patterns

For production use, consider more advanced algorithms:
- **VNG** (Variable Number of Gradients) - Edge-aware interpolation
- **AHD** (Adaptive Homogeneity-Directed) - Better edge preservation
- **DCB** - Modern high-quality algorithm

These algorithms are significantly more complex but reduce color artifacts.

## API Reference

### Main Pipeline

| Function | Description |
|----------|-------------|
| `process_raw_image(filename, ...)` | Complete RAW processing pipeline |
| `read_raw_metadata(filename)` | Extract black level and neutral point |

### Processing Steps

| Function | Description |
|----------|-------------|
| `normalize_black_level(image, black_level)` | Black level correction |
| `demosaic(image, pattern, neutral)` | Full demosaicing pipeline |
| `white_balance(image, white_point)` | Manual white balance |
| `apply_gamma_correction(image, gamma)` | Gamma correction |
| `histogram_stretch(image, low, high)` | Histogram normalization |
| `threshold_binary(image, threshold)` | Binary thresholding |

### Utilities

| Function | Description |
|----------|-------------|
| `save_image(image, filename)` | Save processed image |
