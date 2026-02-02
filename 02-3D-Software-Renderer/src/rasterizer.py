"""
Rasterization algorithms for converting geometry to pixels.

Provides:
- DDA line rasterization
- Barycentric coordinate triangle fill rasterization
"""

import numpy as np
from typing import Tuple

from .mesh import Mesh, Face, Vertex
from .framebuffer import Framebuffer


def rasterize(mesh: Mesh, framebuffer: Framebuffer, mode: str = 'line') -> None:
    """
    Rasterize a mesh to the framebuffer.
    
    Args:
        mesh: Mesh to rasterize (should be in screen space)
        framebuffer: Target framebuffer
        mode: 'line' for wireframe, 'fill' for solid
    """
    framebuffer.clear()
    
    if mode == 'line':
        line_rasterization(mesh, framebuffer)
    elif mode == 'fill':
        fill_rasterization(mesh, framebuffer)
    else:
        raise ValueError(f"Unknown rasterization mode: {mode}")


# =============================================================================
# Line Rasterization (DDA Algorithm)
# =============================================================================

def line_rasterization(mesh: Mesh, framebuffer: Framebuffer) -> None:
    """
    Rasterize mesh as wireframe using DDA line algorithm.
    
    Args:
        mesh: Mesh to rasterize
        framebuffer: Target framebuffer
    """
    for face in mesh.faces:
        vertex_count = face.vertex_count
        for j in range(vertex_count):
            v1 = face.get_vertex(j)
            v2 = face.get_vertex((j + 1) % vertex_count)
            draw_line(framebuffer, v1, v2)


def draw_line(framebuffer: Framebuffer, v1: Vertex, v2: Vertex) -> None:
    """
    Draw a line between two vertices using DDA algorithm.
    
    DDA (Digital Differential Analyzer) interpolates along the line,
    computing pixel positions incrementally.
    
    Args:
        framebuffer: Target framebuffer
        v1: Start vertex
        v2: End vertex
    """
    x1, y1, depth1 = v1.screen_pos
    x2, y2, depth2 = v2.screen_pos

    # Calculate difference
    dx = x2 - x1
    dy = y2 - y1

    # Maximum change in x and y direction - ensures enough points to draw line without gaps
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        # Since v1 and v2 are the same, it doesn't matter which values we use
        framebuffer.set_pixel(
            np.round(x1).astype(int),
            np.round(y1).astype(int),
            depth1,
            v1.color
        )
        return

    # How much to move per step
    # Example: dx = 10, steps = 10, then x_inc = 1, so we move 1 in x direction per step
    x_inc = dx / steps
    y_inc = dy / steps

    # Initialize with start points
    x = x1
    y = y1

    # Set pixels
    for i in range(int(steps)):
        t = i / steps  # Total distance already traveled, where i is steps already taken
        xi = np.round(x).astype(int)  # Round to integer because pixels are discrete
        yi = np.round(y).astype(int)
        color = mix(v1.color, v2.color, t)  # Mix the two colors based on t
        depth = mix(depth1, depth2, t)
        framebuffer.set_pixel(xi, yi, depth, color)  # Set pixel at xi, yi with depth and color
        x += x_inc  # Update position
        y += y_inc


# =============================================================================
# Fill Rasterization (Barycentric Coordinates)
# =============================================================================

def fill_rasterization(mesh: Mesh, framebuffer: Framebuffer) -> None:
    """
    Rasterize mesh as solid triangles using barycentric coordinates.
    
    Handles polygon faces by fan triangulation from first vertex.
    
    Args:
        mesh: Mesh to rasterize
        framebuffer: Target framebuffer
    """
    for face in mesh.faces:
        v1 = face.get_vertex(0)
        # Fan triangulation for polygons with more than 3 vertices
        for j in range(1, face.vertex_count - 1):
            v2 = face.get_vertex(j)
            v3 = face.get_vertex(j + 1)
            draw_triangle(framebuffer, v1, v2, v3)


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation between a and b.
    
    When t = 0, all weight is on a; when t = 1, all weight is on b.
    For example, at t = 0.3 more weight is on a than b: a * 0.7 + b * 0.3
    
    Args:
        a: Start value (t=0)
        b: End value (t=1)
        t: Interpolation factor
        
    Returns:
        Interpolated value
    """
    return a * (1 - t) + b * t


def barycentric_mix(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float
) -> np.ndarray:
    """
    Barycentric interpolation within a triangle.
    
    alpha + beta + gamma = 1
    
    Args:
        a, b, c: Triangle vertex values
        alpha, beta, gamma: Barycentric coordinates
        
    Returns:
        Interpolated value
    """
    return a * alpha + b * beta + c * gamma


def line_eq(a: float, b: float, c: float, x: float, y: float) -> float:
    """
    Evaluate line equation Ax + By + C at point (x, y).
    
    Args:
        a, b, c: Line equation coefficients
        x, y: Point coordinates
        
    Returns:
        Signed distance from point to line
    """
    return a * x + b * y + c


def draw_triangle(
    framebuffer: Framebuffer,
    v1: Vertex,
    v2: Vertex,
    v3: Vertex
) -> None:
    """
    Fill a triangle using barycentric coordinate interpolation.
    
    Args:
        framebuffer: Target framebuffer
        v1, v2, v3: Triangle vertices
    """
    x1, y1, depth1 = v1.screen_pos
    x2, y2, depth2 = v2.screen_pos
    x3, y3, depth3 = v3.screen_pos

    col1 = v1.color
    col2 = v2.color
    col3 = v3.color

    # Calculate triangle area * 2 (cross product)
    a = ((x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1))

    if not np.isclose(a, 0):
        # Swap order of clockwise triangle to make them counter-clockwise
        # If the area is negative, swap points v2 and v3
        if a < 0:
            t = x2
            x2 = x3
            x3 = t

            t = y2
            y2 = y3
            y3 = t

            t = depth2
            depth2 = depth3
            depth3 = t

            t = col2
            col2 = col3
            col3 = t

        # Calculate line parameters for edges using equation Ax + By + C = 0
        edge1_x, edge1_y = x3 - x2, y3 - y2  # Edge v2-v3
        edge2_x, edge2_y = x1 - x3, y1 - y3  # Edge v3-v1
        edge3_x, edge3_y = x2 - x1, y2 - y1  # Edge v1-v2

        # Calculate normal vectors by rotating the edge by 90 degrees
        a1, b1 = -edge1_y, edge1_x
        a2, b2 = -edge2_y, edge2_x
        a3, b3 = -edge3_y, edge3_x

        # Calculate C
        c1 = -(a1 * x2 + b1 * y2)  # Goes through v2, so we use point v2
        c2 = -(a2 * x3 + b2 * y3)
        c3 = -(a3 * x1 + b3 * y1)

        # Calculate bounding box that completely encloses the triangle
        # Defines the smallest and largest x and y values
        # Calculations can be limited to this box
        xmin, xmax = int(min(x1, x2, x3)), int(max(x1, x2, x3))
        ymin, ymax = int(min(y1, y2, y3)), int(max(y1, y2, y3))

        # Check if the box is valid
        if xmax < xmin or ymax < ymin:
            return

        # Create matrices for all x and y values within the box
        # If xmin = 50, xmax = 52, then matrix xs would be:
        # [[50, 51, 52],
        #  [50, 51, 52],
        #  [50, 51, 52]]
        x = np.arange(xmin, xmax + 1)
        y = np.arange(ymin, ymax + 1)
        xs, ys = np.meshgrid(x, y)

        # For each pixel in the box, calculate distance to edge
        # Ax + By + C = 0 describes the triangle edge
        # For a point (x, y), it lies exactly on the line if the equation equals 0
        # If > 0 or < 0, the point is on one side or the other
        p1 = line_eq(a1, b1, c1, xs, ys)
        p2 = line_eq(a2, b2, c2, xs, ys)
        p3 = line_eq(a3, b3, c3, xs, ys)

        # The mask consists of true/false values indicating which points
        # lie inside the triangle
        # Must be <= 0 for all three edges to be inside
        inside_mask = (p1 <= 0) & (p2 <= 0) & (p3 <= 0)
        if not inside_mask.any():
            return

        # Calculate how far the opposite vertex is from an edge
        # E.g., for edge v2-v3, vertex v1 is used
        # The denominator is used later for calculating barycentric coordinates
        # They serve as normalization factors
        denom1 = line_eq(a1, b1, c1, x1, y1)  # Edge v2-v3
        denom2 = line_eq(a2, b2, c2, x2, y2)
        denom3 = line_eq(a3, b3, c3, x3, y3)

        # Avoid division by zero
        # If denom is very small or 0, epsilon is preferred
        # np.sign ensures the original sign is preserved
        epsilon = 1e-8
        denom1 = np.sign(denom1) * max(np.abs(denom1), epsilon)
        denom2 = np.sign(denom2) * max(np.abs(denom2), epsilon)
        denom3 = np.sign(denom3) * max(np.abs(denom3), epsilon)

        # p are the line equation values for each edge
        # alpha indicates how close or far each pixel is to v1 relative to edge v2-v3
        # If a point is exactly on v1, alpha = 1 and beta = gamma = 0
        alpha = p1 / denom1
        beta = p2 / denom2
        gamma = p3 / denom3

        # Since alpha + beta + gamma = 1 must hold, ensure all values are valid
        alpha = np.clip(alpha, 0, 1)
        beta = np.clip(beta, 0, 1)
        gamma = np.clip(gamma, 0, 1)

        # Normalization to guarantee alpha + beta + gamma = 1
        # E.g., alpha = 0.4, beta = 0.3, gamma = 0.2
        # gives total = 0.9
        # alpha = 0.4 / 0.9 = 0.44, beta = 0.3 / 0.9 = 0.33, gamma = 0.22
        # gives new total of 1
        total = alpha + beta + gamma
        total = np.where(total < epsilon, 1.0, total)  # condition, if true, if false
        alpha /= total
        beta /= total
        gamma /= total

        # Interpolate depth and color
        # Combines values weighted with alpha, beta, gamma
        # [..., None] adds a new axis so multiplication can be performed
        depth = barycentric_mix(depth1, depth2, depth3, alpha, beta, gamma)
        color = (alpha[..., None] * col1 + beta[..., None] * col2 + gamma[..., None] * col3)

        # Set pixels
        framebuffer.set_pixel(xs[inside_mask], ys[inside_mask], depth[inside_mask], color[inside_mask])
