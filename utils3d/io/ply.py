import numpy as np

from typing import *
from pathlib import Path


def read_ply(
    file: Union[str, Path],
    encoding: Union[str, None] = None,
    ignore_unknown: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read .ply file, without preprocessing.
    
    Args:
        file (Any): filepath
        encoding (str, optional): 
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: vertices, faces
    """
    import plyfile
    plydata = plyfile.PlyData.read(file)
    vertices = np.stack([plydata['vertex'][k] for k in ['x', 'y', 'z']], axis=-1)
    if 'face' in plydata:
        faces = np.array(plydata['face']['vertex_indices'].tolist())
    else:
        faces = None
    return vertices, faces


def write_ply(
    file: Union[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray = None,
    edges: np.ndarray = None,
    vertex_colors: np.ndarray = None,
    edge_colors: np.ndarray = None,
    text: bool = False
):
    """
    Write .ply file, without preprocessing.
    
    Args:
        file (Any): filepath
        vertices (np.ndarray): [N, 3]
        faces (np.ndarray): [T, E]
        edges (np.ndarray): [E, 2]
        vertex_colors (np.ndarray, optional): [N, 3]. Defaults to None.
        edge_colors (np.ndarray, optional): [E, 3]. Defaults to None.
        text (bool, optional): save data in text format. Defaults to False.
    """
    import plyfile
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    vertices = vertices.astype(np.float32)
    if faces is not None:
        assert faces.ndim == 2
        faces = faces.astype(np.int32)
    if edges is not None:
        assert edges.ndim == 2 and edges.shape[1] == 2
        edges = edges.astype(np.int32)

    if vertex_colors is not None:
        assert vertex_colors.ndim == 2 and vertex_colors.shape[1] == 3
        if vertex_colors.dtype in [np.float32, np.float64]:
            vertex_colors = vertex_colors * 255
        vertex_colors = np.clip(vertex_colors, 0, 255).astype(np.uint8)
        vertices_data = np.zeros(len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices_data['x'] = vertices[:, 0]
        vertices_data['y'] = vertices[:, 1]
        vertices_data['z'] = vertices[:, 2]
        vertices_data['red'] = vertex_colors[:, 0]
        vertices_data['green'] = vertex_colors[:, 1]
        vertices_data['blue'] = vertex_colors[:, 2]
    else:
        vertices_data = np.array([tuple(v) for v in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    if faces is not None:
        faces_data = np.zeros(len(faces), dtype=[('vertex_indices', 'i4', (faces.shape[1],))])
        faces_data['vertex_indices'] = faces

    if edges is not None:
        if edge_colors is not None:
            assert edge_colors.ndim == 2 and edge_colors.shape[1] == 3
            if edge_colors.dtype in [np.float32, np.float64]:
                edge_colors = edge_colors * 255
            edge_colors = np.clip(edge_colors, 0, 255).astype(np.uint8)
            edges_data = np.zeros(len(edges), dtype=[('vertex1', 'i4'), ('vertex2', 'i4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            edges_data['vertex1'] = edges[:, 0]
            edges_data['vertex2'] = edges[:, 1]
            edges_data['red'] = edge_colors[:, 0]
            edges_data['green'] = edge_colors[:, 1]
            edges_data['blue'] = edge_colors[:, 2]
        else:
            edges_data = np.array([tuple(e) for e in edges], dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])
    
    ply_data = [plyfile.PlyElement.describe(vertices_data, 'vertex')]
    if faces is not None:
        ply_data.append(plyfile.PlyElement.describe(faces_data, 'face'))
    if edges is not None:
        ply_data.append(plyfile.PlyElement.describe(edges_data, 'edge'))
    
    plyfile.PlyData(ply_data, text=text).write(file)
    