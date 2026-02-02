"""
PLY file loader for 3D models.

Parses ASCII PLY files and creates Mesh objects with
vertex positions, colors, and face connectivity.
"""

import numpy as np
from typing import Dict, Any

from .mesh import Mesh, Face, Vertex
from .transforms import create_mvp_matrix


class PlyError(RuntimeError):
    """Error during PLY file parsing."""
    pass


def load_ply(filepath: str, apply_transform: bool = True) -> Mesh:
    """
    Load a PLY file and create a Mesh.
    
    Args:
        filepath: Path to the .ply file
        apply_transform: If True, apply default MVP transform
        
    Returns:
        Mesh object with loaded geometry
    """
    data = parse_ply(filepath)
    
    vertex_data = data['vertex']
    face_data = data['face']
    
    # Extract vertex positions (n x 3)
    positions = np.stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ], axis=1)
    
    # Extract vertex colors and normalize to [0, 1]
    colors = np.stack([
        vertex_data['red'],
        vertex_data['green'],
        vertex_data['blue']
    ], axis=1) / 255.0
    
    # Get face indices
    face_indices = face_data['vertex_indices']
    
    # Convert to homogeneous coordinates
    positions = np.hstack([positions, np.ones((positions.shape[0], 1))])
    
    # Apply MVP transform if requested
    if apply_transform:
        mvp = create_mvp_matrix()
        positions = (mvp @ positions.T).T
    
    # Build mesh
    mesh = Mesh()
    for face_idx in face_indices:
        vertices = [
            Vertex(
                position=positions[i].copy(),
                color=colors[i].copy()
            )
            for i in face_idx
        ]
        mesh.add_face(Face(vertices=vertices))
    
    return mesh


def parse_ply(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Parse a PLY file and extract raw data.
    
    Args:
        filepath: Path to the .ply file
        
    Returns:
        Dictionary with 'vertex' and 'face' data
    """
    with open(filepath, 'r') as f:
        lines = iter(f)
        
        # Check magic bytes
        magic = next(lines).strip()
        if magic != 'ply':
            raise PlyError("Not a PLY file")
        
        # Check format
        format_line = next(lines).strip()
        if format_line != 'format ascii 1.0':
            raise PlyError("Only ASCII PLY format is supported")
        
        # Skip comment line
        next(lines)
        
        # Parse header
        elements = []
        dtypes = {
            'float': np.float32,
            'uchar': np.uint8,
            'uint': np.uint32
        }
        
        while True:
            line = next(lines).strip()
            
            if line == 'end_header':
                break
            
            if line.startswith('element'):
                parts = line.split()
                elements.append({
                    'name': parts[1],
                    'count': int(parts[2]),
                    'properties': []
                })
            
            elif line.startswith('property'):
                parts = line.split()
                if not elements:
                    raise PlyError("Property before element definition")
                
                if parts[1] == 'list':
                    elements[-1]['properties'].append({
                        'name': parts[4],
                        'dtype': dtypes[parts[3]],
                        'is_list': True,
                        'list_size': None
                    })
                else:
                    elements[-1]['properties'].append({
                        'name': parts[2],
                        'dtype': dtypes[parts[1]],
                        'is_list': False
                    })
        
        # Parse data
        data = {}
        
        for elem in elements:
            data[elem['name']] = {}
            elem_data = data[elem['name']]
            
            # Initialize arrays for non-list properties
            for prop in elem['properties']:
                if not prop['is_list']:
                    elem_data[prop['name']] = np.empty(
                        elem['count'],
                        dtype=prop['dtype']
                    )
            
            # Read element data
            for i in range(elem['count']):
                line = next(lines)
                values = iter(line.split())
                
                for prop in elem['properties']:
                    if prop['is_list']:
                        count = int(next(values))
                        
                        # Initialize list array on first encounter
                        if prop['name'] not in elem_data:
                            elem_data[prop['name']] = np.empty(
                                (elem['count'], count),
                                dtype=prop['dtype']
                            )
                            prop['list_size'] = count
                        
                        # Read list values
                        for j in range(count):
                            elem_data[prop['name']][i, j] = prop['dtype'](
                                next(values)
                            )
                    else:
                        elem_data[prop['name']][i] = prop['dtype'](
                            next(values)
                        )
        
        return data
