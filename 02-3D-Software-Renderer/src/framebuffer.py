"""
Framebuffer with Z-buffer for rasterization output.

Provides a pixel buffer with depth testing for proper
occlusion handling in 3D rendering.
"""

import numpy as np
from typing import Tuple


class Framebuffer:
    """
    A framebuffer with RGB color buffer and depth buffer.
    
    Attributes:
        width: Width in pixels
        height: Height in pixels
        image: RGB color buffer, shape (height, width, 3)
        zbuffer: Depth buffer, shape (height, width)
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize framebuffer with given dimensions.
        
        Args:
            width: Width in pixels
            height: Height in pixels
        """
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3), dtype=np.float64)
        self.zbuffer = np.ones((height, width), dtype=np.float64)
    
    def set_pixel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        depth: np.ndarray,
        color: np.ndarray
    ) -> None:
        """
        Set pixels with depth testing.
        
        Only pixels with depth less than the current z-buffer value
        will be written. Supports batch operations with arrays.
        
        Args:
            x: X coordinates (int array or scalar)
            y: Y coordinates (int array or scalar)
            depth: Depth values
            color: RGB colors, shape (..., 3)
        """
        x = np.atleast_1d(np.array(x, dtype=int))
        y = np.atleast_1d(np.array(y, dtype=int))
        depth = np.atleast_1d(depth)
        color = np.atleast_2d(color)
        
        # Create coordinate pairs
        coords = np.array(list(zip(x, y)))
        if len(coords) == 0:
            return
        
        # Find pixels that fail depth test
        current_depth = self.zbuffer[coords[:, 1], coords[:, 0]]
        mask = depth < current_depth
        
        if not np.any(mask):
            return
        
        # Filter to only valid pixels
        valid_coords = coords[mask]
        valid_depth = depth[mask]
        valid_color = color[mask] if color.ndim > 1 else color
        
        # Update buffers
        self.zbuffer[valid_coords[:, 1], valid_coords[:, 0]] = valid_depth
        self.image[valid_coords[:, 1], valid_coords[:, 0]] = valid_color
    
    def get_pixel(self, x: int, y: int) -> np.ndarray:
        """
        Get the color value at a specific pixel.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            RGB color value
            
        Raises:
            IndexError: If coordinates are out of bounds
        """
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise IndexError(f"Pixel ({x}, {y}) out of bounds")
        return self.image[y, x]
    
    def clear(self) -> None:
        """Clear the framebuffer to black with maximum depth."""
        self.image.fill(0)
        self.zbuffer.fill(1)
    
    def to_image(self) -> np.ndarray:
        """
        Convert framebuffer to 8-bit RGB image.
        
        Returns:
            RGB image as uint8 array, shape (height, width, 3)
        """
        return np.uint8(np.clip(self.image, 0, 1) * 255)
