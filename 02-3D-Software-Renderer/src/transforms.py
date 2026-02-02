"""
3D transformation matrices for the rendering pipeline.

Provides model, view, and projection matrices for transforming
3D geometry through the graphics pipeline.
"""

import numpy as np


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """
    Create a 4x4 translation matrix.
    
    Args:
        x, y, z: Translation amounts
        
    Returns:
        4x4 translation matrix
    """
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def perspective_matrix(
    fov: float,
    aspect: float,
    near: float,
    far: float
) -> np.ndarray:
    """
    Create a perspective projection matrix.
    
    Args:
        fov: Field of view in degrees
        aspect: Aspect ratio (width / height)
        near: Near clipping plane distance
        far: Far clipping plane distance
        
    Returns:
        4x4 perspective projection matrix
    """
    tan_half = np.tan(np.radians(fov) / 2)
    
    fn1 = -(near + far) / (far - near)
    fn2 = (-2 * far * near) / (far - near)
    
    return np.array([
        [1 / (tan_half * aspect), 0, 0, 0],
        [0, 1 / tan_half, 0, 0],
        [0, 0, fn1, fn2],
        [0, 0, -1, 0]
    ], dtype=np.float64)


def create_mvp_matrix(
    fov: float = 90,
    aspect: float = 1,
    near: float = 0.1,
    far: float = 10,
    camera_z: float = -3
) -> np.ndarray:
    """
    Create a combined Model-View-Projection matrix.
    
    Uses a simple setup with the camera at the origin looking down -Z.
    
    Args:
        fov: Field of view in degrees
        aspect: Aspect ratio
        near: Near clip distance
        far: Far clip distance
        camera_z: Z translation of the model
        
    Returns:
        4x4 MVP matrix
    """
    model = translation_matrix(0, 0, camera_z)
    view = np.eye(4)
    proj = perspective_matrix(fov, aspect, near, far)
    
    return proj @ view @ model
