from io import TextIOWrapper
from typing import Dict, Any, Union, Iterable
import numpy as np
from pathlib import Path

__all__ = [
    'read_obj', 
    'write_obj', 
    'simple_write_obj'
]

def read_obj(
    file : Union[str, Path, TextIOWrapper],
    encoding: Union[str, None] = None,
    ignore_unknown: bool = False
):
    """
    Read wavefront .obj file, without preprocessing.
    
    Why bothering having this read_obj() while we already have other libraries like `trimesh`? 
    This function read the raw format from .obj file and keeps the order of vertices and faces, 
    while trimesh which involves modification like merge/split vertices, which could break the orders of vertices and faces,
    Those libraries are commonly aiming at geometry processing and rendering supporting various formats.
    If you want mesh geometry processing, you may turn to `trimesh` for more features.

    ### Parameters
        `file` (str, Path, TextIOWrapper): filepath or file object
        encoding (str, optional): 
    
    ### Returns
        obj (dict): A dict containing .obj components
        {   
            'mtllib': [],
            'v': [[0,1, 0.2, 1.0], [1.2, 0.0, 0.0], ...],
            'vt': [[0.5, 0.5], ...],
            'vn': [[0., 0.7, 0.7], [0., -0.7, 0.7], ...],
            'f': [[0, 1, 2], [2, 3, 4],...],
            'usemtl': [{'name': 'mtl1', 'f': 7}]
        }
    """
    if hasattr(file,'read'):
        lines = file.read().splitlines()
    else:
        with open(file, 'r', encoding=encoding) as fp:
            lines = fp.read().splitlines()
    mtllib = []
    v, vt, vn, vp = [], [], [], []      # Vertex coordinates, Vertex texture coordinate, Vertex normal, Vertex parameter
    f, ft, fn = [], [], []              # Face indices, Face texture indices, Face normal indices
    o = []
    s = []
    usemtl = []
    
    def pad(l: list, n: Any):
        return l + [n] * (3 - len(l))
    
    for i, line in enumerate(lines):
        sq = line.strip().split()
        if len(sq) == 0: 
            continue
        if sq[0] == 'v':
            assert 4 <= len(sq) <= 5, f'Invalid format of line {i}: {line}'
            v.append([float(e) for e in sq[1:]][:3])
        elif sq[0] == 'vt':
            assert 3 <= len(sq) <= 4, f'Invalid format of line {i}: {line}'
            vt.append([float(e) for e in sq[1:]][:2])
        elif sq[0] == 'vn':
            assert len(sq) == 4, f'Invalid format of line {i}: {line}'
            vn.append([float(e) for e in sq[1:]])
        elif sq[0] == 'vp':
            assert 2 <= len(sq) <= 4, f'Invalid format of line {i}: {line}'
            vp.append(pad([float(e) for e in sq[1:]], 0))
        elif sq[0] == 'f':
            spliting = [pad([int(j) - 1 for j in e.split('/')], -1) for e in sq[1:]]
            f.append([e[0] for e in spliting])
            ft.append([e[1] for e in spliting])
            fn.append([e[2] for e in spliting])
        elif sq[0] == 'usemtl':
            assert len(sq) == 2
            usemtl.append((sq[1], len(f)))
        elif sq[0] == 'o':
            assert len(sq) == 2
            o.append((sq[1], len(f)))
        elif sq[0] == 's':
            s.append((sq[1], len(f)))
        elif sq[0] == 'mtllib':
            assert len(sq) == 2
            mtllib.append(sq[1])
        elif sq[0][0] == '#':
            continue
        else:
            if not ignore_unknown:
                raise Exception(f'Unknown keyword {sq[0]}')
    
    min_poly_vertices = min(len(f) for f in f)
    max_poly_vertices = max(len(f) for f in f)

    return {
        'mtllib': mtllib,
        'v': np.array(v, dtype=np.float32),
        'vt': np.array(vt, dtype=np.float32),
        'vn': np.array(vn, dtype=np.float32),
        'vp': np.array(vp, dtype=np.float32),
        'f': np.array(f, dtype=np.int32) if min_poly_vertices == max_poly_vertices else f,
        'ft': np.array(ft, dtype=np.int32) if min_poly_vertices == max_poly_vertices else ft,
        'fn': np.array(fn, dtype=np.int32) if min_poly_vertices == max_poly_vertices else fn,
        'o': o,
        's': s,
        'usemtl': usemtl,
    }


def write_obj(
        file: Union[str, Path],
        obj: Dict[str, Any],
        encoding: Union[str, None] = None
    ):
    with open(file, 'w', encoding=encoding) as fp:
        for k in ['v', 'vt', 'vn', 'vp']:
            if k not in obj:
                continue
            for v in obj[k]:
                print(k, *map(float, v), file=fp)
        for f in obj['f']:
            print('f', *((str('/').join(map(int, i)) if isinstance(int(i), Iterable) else i) for i in f), file=fp)


def simple_write_obj(
        file: Union[str, Path],
        vertices: np.ndarray,
        faces: np.ndarray,
        encoding: Union[str, None] = None
    ):
    """
    Write wavefront .obj file, without preprocessing.
    
    Args:
        vertices (np.ndarray): [N, 3]
        faces (np.ndarray): [T, 3]
        file (Any): filepath
        encoding (str, optional): 
    """
    with open(file, 'w', encoding=encoding) as fp:
        for v in vertices:
            print('v', *map(float, v), file=fp)
        for f in faces:
            print('f', *map(int, f + 1), file=fp)
