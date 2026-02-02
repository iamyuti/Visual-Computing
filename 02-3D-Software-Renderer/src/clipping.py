"""
Polygon clipping using Sutherland-Hodgman algorithm.

Clips polygons against the canonical view volume (NDC cube)
defined by six clipping planes.
"""

import numpy as np
from typing import List, Optional
from copy import copy

from .mesh import Mesh, Face, Vertex


class ClipPlane:
    """
    A clipping plane in Hessian normal form.
    
    The plane equation is: dot(position, plane) = 0
    Points with dot(position, plane) <= 0 are considered inside.
    """
    
    def __init__(self, plane: np.ndarray):
        """
        Initialize clipping plane.
        
        Args:
            plane: 4D vector [a, b, c, d] defining plane ax + by + cz + d = 0
        """
        self.plane = plane
    
    def is_inside(self, pos: np.ndarray) -> bool:
        """
        Check if a point lies inside (behind) the plane.
        Points on the plane are considered inside.
        
        Args:
            pos: Homogeneous position with 4 components
            
        Returns:
            True if point is inside or on the plane
        """
        # Dot product v * p
        dot_product = np.dot(pos, self.plane)
        
        # Inside if on the plane or behind it
        return dot_product <= 0
    
    def intersect(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Compute intersection of edge between pos1 and pos2 with the plane.
        
        Args:
            pos1: Homogeneous position with 4 components
            pos2: Homogeneous position with 4 components
            
        Returns:
            Normalized intersection value t in [0, 1]
        """
        # Dot product of a (pos1) and p
        l_a = np.dot(pos1, self.plane)
        # Dot product of b (pos2) and p
        l_b = np.dot(pos2, self.plane)

        if l_a == l_b:
            return 0

        # t is the interpolation coefficient
        t = l_a / (l_a - l_b)

        factor = 1e-6
        # If a is inside, reduce t slightly
        if l_a <= 0 < l_b:
            t -= factor
        # If b is inside, increase t slightly
        elif l_b <= 0 < l_a:
            t += factor

        # Clamp t to [0, 1]
        t = max(0.0, min(1.0, t))

        return t


def get_clip_planes() -> List[ClipPlane]:
    """
    Create the six clipping planes for the NDC cube [-1, 1]^3.
    
    The normal vector [a, b, c] points outward with distance d from origin.
    Plane equation: ax + by + cz + d = 0
    
    Returns:
        List of 6 clipping planes
    """
    # For example, for x = -1 (points left) we get -x + d = 0
    # Choosing a point on the plane e.g. (-1, 2, 3)
    # gives d = -1, so the plane is [-1, 0, 0, -1]
    return [
        ClipPlane(np.array([1, 0, 0, -1])),   # Right plane: x = 1
        ClipPlane(np.array([-1, 0, 0, -1])),  # Left plane: x = -1
        ClipPlane(np.array([0, 1, 0, -1])),   # Top plane: y = 1
        ClipPlane(np.array([0, -1, 0, -1])),  # Bottom plane: y = -1
        ClipPlane(np.array([0, 0, 1, -1])),   # Far plane: z = 1 (away from viewer)
        ClipPlane(np.array([0, 0, -1, -1]))   # Near plane: z = -1 (towards viewer)
    ]


def clip_polygon_against_plane(
    vertex_count: int,
    positions: np.ndarray,
    colors: np.ndarray,
    plane: ClipPlane
) -> tuple:
    """
    Clip polygon against a single plane using Sutherland-Hodgman algorithm.
    
    Args:
        vertex_count: Number of vertices in the polygon
        positions: n x 4 matrix with vertex positions
        colors: n x 3 matrix with vertex colors
        plane: Clipping plane
        
    Returns:
        Tuple of (vertex_count_clipped, pos_clipped, col_clipped)
    """
    # +1 because clipping can create an additional point
    pos_clipped = np.zeros((vertex_count + 1, 4))
    col_clipped = np.zeros((vertex_count + 1, 3))
    vertex_count_clipped = 0

    # Iterate through all vertices from 0 to vertex_count - 1, examining edges
    for i in range(vertex_count):
        curr_pos = positions[i]  # End point
        curr_col = colors[i]

        # Modulo for closed polygon, since the endpoint is always connected to the start point
        # If polygon is a triangle and i = 0, we get the predecessor with -1 % 3 = 2
        prev_index = (i - 1) % vertex_count
        prev_pos = positions[prev_index]  # Start point
        prev_col = colors[prev_index]

        inside_curr = plane.is_inside(curr_pos)  # True if current point is inside
        inside_prev = plane.is_inside(prev_pos)  # True if previous point is inside

        # Case 1: Current point is inside
        if inside_curr:
            # Case 1.1: Previous point is outside, edge goes from outside to inside
            if not inside_prev:
                t = plane.intersect(prev_pos, curr_pos)  # Calculate parameter t (0 = prev, 1 = curr)
                inter_pos = prev_pos * (1 - t) + curr_pos * t  # Interpolated position
                inter_col = prev_col * (1 - t) + curr_col * t  # Interpolated color

                # Add intersection point
                pos_clipped[vertex_count_clipped] = inter_pos
                col_clipped[vertex_count_clipped] = inter_col
                vertex_count_clipped += 1

            # Current point is always added because it's inside
            pos_clipped[vertex_count_clipped] = curr_pos
            col_clipped[vertex_count_clipped] = curr_col
            vertex_count_clipped += 1

        # Case 2: Previous point is inside but current is not, edge goes outward
        elif inside_prev:
            t = plane.intersect(prev_pos, curr_pos)
            inter_pos = prev_pos * (1 - t) + curr_pos * t
            inter_col = prev_col * (1 - t) + curr_col * t

            pos_clipped[vertex_count_clipped] = inter_pos
            col_clipped[vertex_count_clipped] = inter_col
            vertex_count_clipped += 1

    # Trim arrays to actual number of used points
    pos_clipped = pos_clipped[:vertex_count_clipped]
    col_clipped = col_clipped[:vertex_count_clipped]

    return vertex_count_clipped, pos_clipped, col_clipped


def clip_mesh(mesh: Mesh, planes: Optional[List[ClipPlane]] = None) -> Mesh:
    """
    Clip all faces of a mesh against the view volume.
    
    Args:
        mesh: Mesh to clip
        planes: Clipping planes (defaults to NDC cube planes)
        
    Returns:
        New mesh with clipped faces
    """
    if planes is None:
        planes = get_clip_planes()
    
    clipped_mesh = Mesh()
    
    for face in mesh.faces:
        # Extract positions and colors from face vertices
        positions = np.array([v.position for v in face.vertices])
        colors = np.array([v.color for v in face.vertices])
        vertex_count = len(face.vertices)
        
        # Clip against each plane
        for plane in planes:
            vertex_count, positions, colors = clip_polygon_against_plane(
                vertex_count, positions, colors, plane
            )
            if vertex_count == 0:
                break
        
        # Add clipped face if it has vertices remaining
        if vertex_count > 0:
            vertices = [
                Vertex(position=positions[i].copy(), color=colors[i].copy())
                for i in range(vertex_count)
            ]
            clipped_mesh.add_face(Face(vertices=vertices))
    
    return clipped_mesh
