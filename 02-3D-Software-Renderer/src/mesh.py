"""
Mesh data structures for 3D geometry representation.

Provides Vertex and Mesh classes for storing and manipulating
3D model data with support for homogeneous coordinates and
screen space transformation.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Vertex:
    """
    A vertex with homogeneous position and RGB color.
    
    Attributes:
        position: 4D homogeneous coordinates (x, y, z, w)
        color: RGB color values in range [0, 1]
        screen_pos: Screen coordinates after projection (x, y, depth)
    """
    position: np.ndarray
    color: np.ndarray
    screen_pos: Optional[np.ndarray] = None
    
    @staticmethod
    def interpolate(a: 'Vertex', b: 'Vertex', t: float) -> 'Vertex':
        """
        Linear interpolation between two vertices.
        
        Args:
            a: Start vertex (t=0)
            b: End vertex (t=1)
            t: Interpolation factor in [0, 1]
            
        Returns:
            Interpolated vertex
        """
        # Linear interpolation: result = a * (1 - t) + b * t
        position = a.position * (1 - t) + b.position * t
        color = a.color * (1 - t) + b.color * t
        return Vertex(position=position, color=color)
    
    @staticmethod
    def barycentric_interpolate(
        a: 'Vertex', b: 'Vertex', c: 'Vertex',
        alpha: float, beta: float, gamma: float
    ) -> 'Vertex':
        """
        Barycentric interpolation within a triangle.
        
        Args:
            a, b, c: Triangle vertices
            alpha, beta, gamma: Barycentric coordinates (should sum to 1)
            
        Returns:
            Interpolated vertex
        """
        # Barycentric interpolation: result = a * alpha + b * beta + c * gamma
        position = a.position * alpha + b.position * beta + c.position * gamma
        color = a.color * alpha + b.color * beta + c.color * gamma
        return Vertex(position=position, color=color)


@dataclass
class Face:
    """
    A polygon face consisting of multiple vertices.
    
    Starts as triangles but may have more vertices after clipping
    (up to 9 for triangle clipped against 6 planes).
    """
    vertices: List[Vertex] = field(default_factory=list)
    
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)
    
    def get_vertex(self, index: int) -> Vertex:
        return self.vertices[index]


class Mesh:
    """
    A 3D mesh consisting of polygon faces.
    
    Supports transformation, homogenization, and screen space conversion.
    """
    
    def __init__(self, faces: Optional[List[Face]] = None):
        self.faces: List[Face] = faces if faces is not None else []
    
    @property
    def face_count(self) -> int:
        return len(self.faces)
    
    @property
    def vertex_count(self) -> int:
        return sum(face.vertex_count for face in self.faces)
    
    def add_face(self, face: Face) -> None:
        """Add a face to the mesh."""
        self.faces.append(face)
    
    def transform(self, matrix: np.ndarray) -> None:
        """
        Apply a 4x4 transformation matrix to all vertices.
        
        Args:
            matrix: 4x4 transformation matrix (e.g., MVP matrix)
        """
        for face in self.faces:
            for vertex in face.vertices:
                vertex.position = matrix @ vertex.position
    
    def homogenize(self) -> None:
        """
        Perform perspective division on all vertices.
        
        Divides each position by its w component to convert
        from clip space to normalized device coordinates.
        """
        for face in self.faces:
            for vertex in face.vertices:
                w = vertex.position[3]
                if w != 0:
                    vertex.position = vertex.position / w
    
    def to_screen_space(self, width: int, height: int) -> None:
        """
        Transform vertices from NDC [-1, 1] to screen coordinates.
        
        Args:
            width: Framebuffer width in pixels
            height: Framebuffer height in pixels
        """
        eps = 0.001
        sx = (width - eps) / 2.0
        sy = (height - eps) / -2.0  # Flip Y axis
        dx = (width - eps) / 2.0
        dy = (height - eps) / 2.0
        
        for face in self.faces:
            for vertex in face.vertices:
                x = np.floor(vertex.position[0] * sx + dx)
                y = np.floor(vertex.position[1] * sy + dy)
                z = vertex.position[2]
                vertex.screen_pos = np.array([x, y, z])
    
    def copy(self) -> 'Mesh':
        """Create a deep copy of the mesh."""
        new_faces = []
        for face in self.faces:
            new_vertices = [
                Vertex(
                    position=v.position.copy(),
                    color=v.color.copy(),
                    screen_pos=v.screen_pos.copy() if v.screen_pos is not None else None
                )
                for v in face.vertices
            ]
            new_faces.append(Face(vertices=new_vertices))
        return Mesh(faces=new_faces)
