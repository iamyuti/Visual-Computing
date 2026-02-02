#!/usr/bin/env python3
"""
3D Software Rasterizer - Command Line Interface

A from-scratch implementation of a software rasterizer demonstrating
core computer graphics algorithms including Sutherland-Hodgman clipping,
DDA line rasterization, and barycentric fill rasterization.

Usage:
    python main.py --model data/star.ply --mode fill --output star.png
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.ply_loader import load_ply
from src.clipping import clip_mesh
from src.rasterizer import rasterize
from src.framebuffer import Framebuffer


def main():
    parser = argparse.ArgumentParser(
        description='3D Software Rasterizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --model data/star.ply --mode line
    python main.py --model data/torus.ply --mode fill --output torus.png
    python main.py --model data/star.ply --mode fill --width 800 --height 800
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        default='data/star.ply',
        help='Path to the .ply model file'
    )
    parser.add_argument(
        '--mode',
        choices=['line', 'fill'],
        default='line',
        help='Rasterization mode: line (wireframe) or fill (solid)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output image path (default: <model_name>_<mode>.png)'
    )
    parser.add_argument(
        '--width', '-W',
        type=int,
        default=600,
        help='Framebuffer width in pixels'
    )
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=600,
        help='Framebuffer height in pixels'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Generate default output path
    if args.output is None:
        model_name = Path(args.model).stem
        args.output = f"examples/{model_name}_{args.mode}.png"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading model: {args.model}")
    mesh = load_ply(args.model)
    print(f"  Faces: {mesh.face_count}")
    print(f"  Vertices: {mesh.vertex_count}")
    
    print("Clipping against view volume...")
    clipped_mesh = clip_mesh(mesh)
    print(f"  Clipped faces: {clipped_mesh.face_count}")
    
    print("Transforming to screen space...")
    clipped_mesh.homogenize()
    clipped_mesh.to_screen_space(args.width, args.height)
    
    print(f"Rasterizing ({args.mode} mode)...")
    framebuffer = Framebuffer(args.width, args.height)
    rasterize(clipped_mesh, framebuffer, mode=args.mode)
    
    print(f"Saving image: {args.output}")
    image = framebuffer.to_image()
    Image.fromarray(image).save(args.output)
    
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
