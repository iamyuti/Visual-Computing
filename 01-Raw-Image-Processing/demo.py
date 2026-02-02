#!/usr/bin/env python3
"""
Demo Script for RAW Image Processing Pipeline
==============================================

This script demonstrates how to use the pipeline to process
a RAW camera image and save the result.

Usage:
    python demo.py
    python demo.py path/to/your/image.dng
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    process_raw_image,
    save_image,
    read_raw_metadata,
    normalize_black_level,
    demosaic,
    apply_gamma_correction,
    histogram_stretch,
    load_image
)
import numpy as np
from PIL import Image


def demo_full_pipeline(input_path: str, output_path: str = None):
    """
    Demonstrate the complete RAW processing pipeline.
    
    Args:
        input_path: Path to RAW image file
        output_path: Optional output path (default: results/input_processed.png)
    """
    print(f"Processing: {input_path}")
    print("-" * 50)
    
    # Process with default settings
    result = process_raw_image(
        input_path,
        bayer_pattern='RGGB',
        gamma=2.2
    )
    
    # Generate output path if not specified (save to results folder)
    if output_path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        base = Path(input_path).stem
        output_path = results_dir / f"{base}_processed.png"
    
    # Save result
    save_image(result, output_path)
    print(f"✓ Saved result to: {output_path}")
    print(f"  Image size: {result.shape[1]}x{result.shape[0]} pixels")
    print(f"  Value range: [{result.min():.3f}, {result.max():.3f}]")
    
    return result


def demo_step_by_step(input_path: str):
    """
    Demonstrate each processing step individually.
    
    Args:
        input_path: Path to RAW image file
    """
    print(f"\nStep-by-step processing: {input_path}")
    print("=" * 50)
    
    # Step 1: Load and read metadata
    print("\n[1/5] Reading metadata...")
    black_level, neutral = read_raw_metadata(input_path)
    print(f"      Black level: {black_level}")
    print(f"      Neutral: {neutral}")
    
    # Load raw image
    img = Image.open(input_path)
    raw = np.array(img)
    print(f"      Raw image shape: {raw.shape}")
    print(f"      Raw value range: [{raw.min()}, {raw.max()}]")
    
    # Step 2: Black level correction
    print("\n[2/5] Applying black level correction...")
    normalized = normalize_black_level(raw, black_level)
    print(f"      Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Step 3: Demosaicing
    print("\n[3/5] Demosaicing (RGGB pattern)...")
    rgb = demosaic(normalized, pattern='RGGB', neutral=neutral)
    print(f"      RGB shape: {rgb.shape}")
    print(f"      RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    # Step 4: Histogram stretching
    print("\n[4/5] Stretching histogram...")
    rgb = histogram_stretch(rgb)
    print(f"      Stretched range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    # Step 5: Gamma correction
    print("\n[5/5] Applying gamma correction (γ=2.2)...")
    rgb = apply_gamma_correction(rgb, gamma=2.2)
    print(f"      Final range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    # Clip and save to results folder
    result = np.clip(rgb, 0, 1)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    base = Path(input_path).stem
    output_path = results_dir / f"{base}_step_by_step.png"
    save_image(result, str(output_path))
    print(f"\n✓ Saved to: {output_path}")
    
    return result


def main():
    """Main entry point."""
    # Determine input file
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Use example image if available
        example_path = Path(__file__).parent / "examples" / "IMG_0.tiff"
        if example_path.exists():
            input_path = str(example_path)
        else:
            print("Usage: python demo.py <path_to_raw_image>")
            print("\nNo example image found. Please provide a RAW image path.")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Run demos
    print("=" * 50)
    print("RAW Image Processing Pipeline - Demo")
    print("=" * 50)
    
    # Demo 1: Full pipeline (one-liner)
    demo_full_pipeline(input_path)
    
    # Demo 2: Step-by-step processing
    demo_step_by_step(input_path)
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
