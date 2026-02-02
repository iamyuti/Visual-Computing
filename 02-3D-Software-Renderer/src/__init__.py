"""
3D Software Rasterizer

A from-scratch implementation demonstrating core computer graphics algorithms:
- Sutherland-Hodgman polygon clipping
- DDA line rasterization
- Barycentric coordinate fill rasterization
- Z-buffer depth testing
"""

from .mesh import Mesh, Vertex
from .framebuffer import Framebuffer
from .clipping import clip_mesh, ClipPlane
from .rasterizer import rasterize
from .ply_loader import load_ply
