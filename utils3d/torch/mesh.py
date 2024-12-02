import torch
import torch.nn.functional as F
from typing import *
from ._helpers import batched


__all__ = [
    'triangulate',
    'compute_face_normal',
    'compute_face_angles',
    'compute_vertex_normal',
    'compute_vertex_normal_weighted',
    'compute_edges',
    'compute_connected_components',
    'compute_edge_connected_components',
    'compute_boundarys',
    'compute_dual_graph',
    'remove_unreferenced_vertices',
    'remove_corrupted_faces',
    'remove_isolated_pieces',
    'merge_duplicate_vertices',
    'subdivide_mesh_simple',
    'compute_face_tbn',
    'compute_vertex_tbn',
    'laplacian',
    'laplacian_smooth_mesh',
    'taubin_smooth_mesh',
    'laplacian_hc_smooth_mesh',
]


def _group(
    values: torch.Tensor,
    required_group_size: Optional[int] = None,
    return_values: bool = False
) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
    """
    Group values into groups with identical values.
    
    Args:
        values (torch.Tensor): [N] values to group
        required_group_size (int, optional): required group size. Defaults to None.
        return_values (bool, optional): return values of groups. Defaults to False.
        
    Returns:
        group (Union[List[torch.Tensor], torch.Tensor]): list of groups or group indices. It will be a list of groups if required_group_size is None, otherwise a tensor of group indices.
        group_values (Optional[torch.Tensor]): values of groups. Only returned if return_values is True.
    """
    sorted_values, indices = torch.sort(values)
    nondupe = torch.cat([torch.tensor([True], dtype=torch.bool, device=values.device), sorted_values[1:] != sorted_values[:-1]])
    nondupe_indices = torch.cumsum(nondupe, dim=0) - 1
    counts = torch.bincount(nondupe_indices)
    if required_group_size is None:
        groups = torch.split(indices, counts.tolist())
        if return_values:
            group_values = sorted_values[nondupe]
            return groups, group_values
        else:
            return groups
    else:
        counts = counts[nondupe_indices]
        groups = indices[counts == required_group_size].reshape(-1, required_group_size)
        if return_values:
            group_values = sorted_values[nondupe][counts[nondupe] == required_group_size]
            return groups, group_values
        else:
            return groups

def triangulate(
    faces: torch.Tensor,
    vertices: torch.Tensor = None,
    backslash: bool = None
) -> torch.Tensor:
    """
    Triangulate a polygonal mesh.

    Args:
        faces (torch.Tensor): [..., L, P] polygonal faces
        vertices (torch.Tensor, optional): [..., N, 3] 3-dimensional vertices.
            If given, the triangulation is performed according to the distance
            between vertices. Defaults to None.
        backslash (torch.Tensor, optional): [..., L] boolean array indicating
            how to triangulate the quad faces. Defaults to None.


    Returns:
        (torch.Tensor): [L * (P - 2), 3] triangular faces
    """
    if faces.shape[-1] == 3:
        return faces
    P = faces.shape[-1]
    if vertices is not None:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        if backslash is None:
            faces_idx = faces.long()
            backslash = torch.norm(vertices[faces_idx[..., 0]] - vertices[faces_idx[..., 2]], p=2, dim=-1) < \
                        torch.norm(vertices[faces_idx[..., 1]] - vertices[faces_idx[..., 3]], p=2, dim=-1)
    if backslash is None:
        loop_indice = torch.stack([
            torch.zeros(P - 2, dtype=int),
            torch.arange(1, P - 1, 1, dtype=int),
            torch.arange(2, P, 1, dtype=int)
        ], axis=1)
        return faces[:, loop_indice].reshape(-1, 3)
    else:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        if isinstance(backslash, bool):
            if backslash:
                faces = faces[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
            else:
                faces = faces[:, [0, 1, 3, 3, 1, 2]].reshape(-1, 3)
        else:
            faces = torch.where(
                backslash[:, None],
                faces[:, [0, 1, 2, 0, 2, 3]],
                faces[:, [0, 1, 3, 3, 1, 2]]
            ).reshape(-1, 3)
        return faces


@batched(2, None)
def compute_face_normal(
    vertices: torch.Tensor,
    faces: torch.Tensor
) -> torch.Tensor:
    """
    Compute face normals of a triangular mesh

    Args:
        vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
        faces (torch.Tensor): [..., T, 3] triangular face indices

    Returns:
        normals (torch.Tensor): [..., T, 3] face normals
    """
    N = vertices.shape[0]
    index = torch.arange(N)[:, None]
    normal = torch.cross(
        vertices[index, faces[..., 1].long()] - vertices[index, faces[..., 0].long()],
        vertices[index, faces[..., 2].long()] - vertices[index, faces[..., 0].long()],
        dim=-1
    )
    return F.normalize(normal, p=2, dim=-1)


@batched(2, None)
def compute_face_angles(
    vertices: torch.Tensor,
    faces: torch.Tensor
) -> torch.Tensor:
    """
    Compute face angles of a triangular mesh

    Args:
        vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices

    Returns:
        angles (torch.Tensor): [..., T, 3] face angles
    """
    face_angles = []
    for i in range(3):
        edge1 = torch.index_select(vertices, dim=-2, index=faces[:, (i + 1) % 3]) - torch.index_select(vertices, dim=-2, index=faces[:, i])
        edge2 = torch.index_select(vertices, dim=-2, index=faces[:, (i + 2) % 3]) - torch.index_select(vertices, dim=-2, index=faces[:, i])
        face_angle = torch.arccos(torch.sum(F.normalize(edge1, p=2, dim=-1) * F.normalize(edge2, p=2, dim=-1), dim=-1))
        face_angles.append(face_angle)
    face_angles = torch.stack(face_angles, dim=-1)
    return face_angles


@batched(2, None, 2)
def compute_vertex_normal(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normal: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute vertex normals of a triangular mesh by averaging neightboring face normals

    Args:
        vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices
        face_normal (torch.Tensor, optional): [..., T, 3] face normals.
            None to compute face normals from vertices and faces. Defaults to None.

    Returns:
        normals (torch.Tensor): [..., N, 3] vertex normals
    """
    N = vertices.shape[0]
    assert faces.shape[-1] == 3, "Only support triangular mesh"
    if face_normal is None:
        face_normal = compute_face_normal(vertices, faces)
    face_normal = face_normal[:, :, None, :].expand(-1, -1, 3, -1).flatten(-3, -2)
    faces = faces.flatten()
    vertex_normal = torch.index_put(torch.zeros_like(vertices), (torch.arange(N)[:, None], faces[None, :]), face_normal, accumulate=True)
    vertex_normal = F.normalize(vertex_normal, p=2, dim=-1)
    return vertex_normal


@batched(2, None, 2)
def compute_vertex_normal_weighted(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normal: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute vertex normals of a triangular mesh by weighted sum of neightboring face normals
    according to the angles

    Args:
        vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices
        face_normal (torch.Tensor, optional): [..., T, 3] face normals.
            None to compute face normals from vertices and faces. Defaults to None.

    Returns:
        normals (torch.Tensor): [..., N, 3] vertex normals
    """
    N = vertices.shape[0]
    if face_normal is None:
        face_normal = compute_face_normal(vertices, faces)
    face_angle = compute_face_angles(vertices, faces)
    face_normal = face_normal[:, :, None, :].expand(-1, -1, 3, -1) * face_angle[..., None]
    vertex_normal = torch.index_put(torch.zeros_like(vertices), (torch.arange(N)[:, None], faces.view(N, -1)), face_normal.view(N, -1, 3), accumulate=True)
    vertex_normal = F.normalize(vertex_normal, p=2, dim=-1)
    return vertex_normal


def compute_edges(
    faces: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute edges of a mesh.
    
    Args:
        faces (torch.Tensor): [T, 3] triangular face indices
        
    Returns:
        edges (torch.Tensor): [E, 2] edge indices
        face2edge (torch.Tensor): [T, 3] mapping from face to edge
        counts (torch.Tensor): [E] degree of each edge
    """
    T = faces.shape[0]
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)    # [3T, 2]
    edges = torch.sort(edges, dim=1).values
    edges, inv_map, counts = torch.unique(edges, return_inverse=True, return_counts=True, dim=0)
    face2edge = inv_map.view(3, T).T
    return edges, face2edge, counts


def compute_connected_components(
    faces: torch.Tensor,
    edges: torch.Tensor=None,
    face2edge: torch.Tensor=None
) -> List[torch.Tensor]:
    """
    Compute connected faces of a mesh.

    Args:
        faces (torch.Tensor): [T, 3] triangular face indices
        edges (torch.Tensor, optional): [E, 2] edge indices. Defaults to None.
        face2edge (torch.Tensor, optional): [T, 3] mapping from face to edge. Defaults to None.
            NOTE: If edges and face2edge are not provided, they will be computed.

    Returns:
        components (List[torch.Tensor]): list of connected faces
    """
    T = faces.shape[0]
    if edges is None or face2edge is None:
        edges, face2edge, _ = compute_edges(faces)
    E = edges.shape[0]

    labels = torch.arange(T, dtype=torch.int32, device=faces.device)
    while True:
        edge_labels = torch.scatter_reduce(
            torch.zeros(E, dtype=torch.int32, device=faces.device),
            0,
            face2edge.flatten().long(),
            labels.view(-1, 1).expand(-1, 3).flatten(),
            reduce='amin',
            include_self=False
        )
        new_labels = torch.min(edge_labels[face2edge], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components = _group(labels)

    return components


def compute_edge_connected_components(
    edges: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Compute connected edges of a mesh.

    Args:
        edges (torch.Tensor): [E, 2] edge indices

    Returns:
        components (List[torch.Tensor]): list of connected edges
    """
    E = edges.shape[0]

    # Re-index edges
    verts, edges = torch.unique(edges.flatten(), return_inverse=True)
    edges = edges.view(-1, 2)
    V = verts.shape[0]

    labels = torch.arange(E, dtype=torch.int32, device=edges.device)
    while True:
        vertex_labels = torch.scatter_reduce(
            torch.zeros(V, dtype=torch.int32, device=edges.device),
            0,
            edges.flatten().long(),
            labels.view(-1, 1).expand(-1, 2).flatten(),
            reduce='amin',
            include_self=False
        )
        new_labels = torch.min(vertex_labels[edges], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components = _group(labels)

    return components


def compute_boundarys(
    faces: torch.Tensor,
    edges: torch.Tensor=None,
    face2edge: torch.Tensor=None,
    edge_degrees: torch.Tensor=None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute boundary edges of a mesh.

    Args:
        faces (torch.Tensor): [T, 3] triangular face indices
        edges (torch.Tensor): [E, 2] edge indices.
        face2edge (torch.Tensor): [T, 3] mapping from face to edge.
        edge_degrees (torch.Tensor): [E] degree of each edge.

    Returns:
        boundary_edge_indices (List[torch.Tensor]): list of boundary edge indices
        boundary_face_indices (List[torch.Tensor]): list of boundary face indices
    """    
    # Map each edge to boundary edge index
    boundary_edges = edges[edge_degrees == 1]                                                                     # [BE, 2]
    boundary_edges_idx = torch.nonzero(edge_degrees == 1, as_tuple=False).flatten()                               # [BE]
    E = edges.shape[0]                                                                                             # Edge count
    BE = boundary_edges.shape[0]                                                                            # Boundary edge count
    map_to_boundary_edges = torch.full((E,), -1, dtype=torch.int32, device=faces.device)                    # [E]
    map_to_boundary_edges[boundary_edges_idx] = torch.arange(BE, dtype=torch.int32, device=faces.device)
    
    # Re-index boundary vertices
    boundary_vertices, boundary_edges = torch.unique(boundary_edges.flatten(), return_inverse=True)
    boundary_edges = boundary_edges.view(-1, 2)
    BV = boundary_vertices.shape[0]
    
    boundary_edge_labels = torch.arange(BE, dtype=torch.int32, device=faces.device)
    while True:
        boundary_vertex_labels = torch.scatter_reduce(
            torch.zeros(BV, dtype=torch.int32, device=faces.device),
            0,
            boundary_edges.flatten().long(),
            boundary_edge_labels.view(-1, 1).expand(-1, 2).flatten(),
            reduce='amin',
            include_self=False
        )
        new_boundary_edge_labels = torch.min(boundary_vertex_labels[boundary_edges], dim=-1).values
        if torch.equal(boundary_edge_labels, new_boundary_edge_labels):
            break
        boundary_edge_labels = new_boundary_edge_labels
        
    labels = torch.unique(boundary_edge_labels)
    boundary_edge_indices = [boundary_edges_idx[boundary_edge_labels == label] for label in labels]
    edge_labels = torch.full((E,), -1, dtype=torch.int32, device=faces.device)
    edge_labels[boundary_edges_idx] = boundary_edge_labels
    boundary_face_indices = [torch.nonzero((edge_labels[face2edge] == label).any(dim=-1), as_tuple=False).flatten() for label in labels]
    
    return boundary_edge_indices, boundary_face_indices


def compute_dual_graph(
    face2edge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dual graph of a mesh.
    
    Args:
        face2edge (torch.Tensor): [T, 3] mapping from face to edge.
            
    Returns:
        dual_edges (torch.Tensor): [DE, 2] face indices of dual edges
        dual_edge2edge (torch.Tensor): [DE] mapping from dual edge to edge
    """
    all_edge_indices = face2edge.flatten()  # [3T]
    dual_edges, dual_edge2edge = _group(all_edge_indices, required_group_size=2, return_values=True)
    dual_edges = dual_edges // face2edge.shape[1]
    return dual_edges, dual_edge2edge


def remove_unreferenced_vertices(
    faces: torch.Tensor,
    *vertice_attrs,
    return_indices: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Remove unreferenced vertices of a mesh. 
    Unreferenced vertices are removed, and the face indices are updated accordingly.

    Args:
        faces (torch.Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes

    Returns:
        faces (torch.Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (torch.Tensor, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = torch.unique(faces, return_inverse=True)
    faces = inv_map.to(torch.int32).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)


def remove_corrupted_faces(
    faces: torch.Tensor
) -> torch.Tensor:
    """
    Remove corrupted faces (faces with duplicated vertices)

    Args:
        faces (torch.Tensor): [T, 3] triangular face indices

    Returns:
        torch.Tensor: [T_, 3] triangular face indices
    """
    corrupted = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    return faces[~corrupted]


def merge_duplicate_vertices(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge duplicate vertices of a triangular mesh. 
    Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

    Args:
        vertices (torch.Tensor): [N, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices
        tol (float, optional): tolerance for merging. Defaults to 1e-6.

    Returns:
        vertices (torch.Tensor): [N_, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices
    """
    vertices_round = torch.round(vertices / tol)
    uni, uni_inv = torch.unique(vertices_round, dim=0, return_inverse=True)
    uni[uni_inv] = vertices
    faces = uni_inv[faces]
    return uni, faces


def remove_isolated_pieces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    connected_components: List[torch.Tensor] = None,
    thresh_num_faces: int = None,
    thresh_radius: float = None,
    thresh_boundary_ratio: float = None,
    remove_unreferenced: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove isolated pieces of a mesh. 
    Isolated pieces are removed, and the face indices are updated accordingly.
    If no face is left, will return the largest connected component.

    Args:
        vertices (torch.Tensor): [N, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices
        connected_components (List[torch.Tensor], optional): connected components of the mesh. If None, it will be computed. Defaults to None.
        thresh_num_faces (int, optional): threshold of number of faces for isolated pieces. Defaults to None.
        thresh_radius (float, optional): threshold of radius for isolated pieces. Defaults to None.
        remove_unreferenced (bool, optional): remove unreferenced vertices after removing isolated pieces. Defaults to True.

    Returns:
        vertices (torch.Tensor): [N_, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices
    """
    if connected_components is None:
        connected_components = compute_connected_components(faces)
    connected_components = sorted(connected_components, key=lambda x: len(x), reverse=True)
    if thresh_num_faces is not None:
        removed = []
        for i in range(1, len(connected_components)):
            if len(connected_components[i]) < thresh_num_faces:
                removed.append(i)
        for i in removed[::-1]:
            connected_components.pop(i)
    if thresh_radius is not None:
        removed = []
        for i in range(1, len(connected_components)):
            comp_vertices = vertices[faces[connected_components[i]].flatten().unique()]
            comp_center = comp_vertices.mean(dim=0)
            comp_radius = (comp_vertices - comp_center).norm(p=2, dim=-1).max()
            if comp_radius < thresh_radius:
                removed.append(i)
        for i in removed[::-1]:
            connected_components.pop(i)
    if thresh_boundary_ratio is not None:
        removed = []
        for i in range(1, len(connected_components)):
            edges = torch.cat([faces[connected_components[i]][:, [0, 1]], faces[connected_components[i]][:, [1, 2]], faces[connected_components[i]][:, [2, 0]]], dim=0)
            edges = torch.sort(edges, dim=1).values
            edges, counts = torch.unique(edges, return_counts=True, dim=0)
            num_boundary_edges = (counts == 1).sum().item()
            num_faces = len(connected_components[i])
            if num_boundary_edges / num_faces > thresh_boundary_ratio:
                removed.append(i)
        for i in removed[::-1]:
            connected_components.pop(i)
    
    # post-process
    faces = torch.cat([faces[connected_components[i]] for i in range(len(connected_components))], dim=0)
    if remove_unreferenced:
        faces, vertices = remove_unreferenced_vertices(faces, vertices)
    return vertices, faces


def subdivide_mesh_simple(vertices: torch.Tensor, faces: torch.Tensor, n: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
    NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.
    
    Args:
        vertices (torch.Tensor): [N, 3] 3-dimensional vertices
        faces (torch.Tensor): [T, 3] triangular face indices
        n (int, optional): number of subdivisions. Defaults to 1.

    Returns:
        vertices (torch.Tensor): [N_, 3] subdivided 3-dimensional vertices
        faces (torch.Tensor): [4 * T, 3] subdivided triangular face indices
    """
    for _ in range(n):
        edges = torch.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=2)
        uni_edges, uni_inv = torch.unique(edges, return_inverse=True, dim=0)
        midpoints = (vertices[uni_edges[:, 0]] + vertices[uni_edges[:, 1]]) / 2

        n_vertices = vertices.shape[0]
        vertices = torch.cat([vertices, midpoints], dim=0)
        faces = torch.cat([
            torch.stack([faces[:, 0], n_vertices + uni_inv[0], n_vertices + uni_inv[2]], axis=1),
            torch.stack([faces[:, 1], n_vertices + uni_inv[1], n_vertices + uni_inv[0]], axis=1),
            torch.stack([faces[:, 2], n_vertices + uni_inv[2], n_vertices + uni_inv[1]], axis=1),
            torch.stack([n_vertices + uni_inv[0], n_vertices + uni_inv[1], n_vertices + uni_inv[2]], axis=1),
        ], dim=0)
    return vertices, faces


def compute_face_tbn(pos: torch.Tensor, faces_pos: torch.Tensor, uv: torch.Tensor, faces_uv: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """compute TBN matrix for each face

    Args:
        pos (torch.Tensor): shape (..., N_pos, 3), positions
        faces_pos (torch.Tensor): shape(T, 3) 
        uv (torch.Tensor): shape (..., N_uv, 3) uv coordinates, 
        faces_uv (torch.Tensor): shape(T, 3) 
        
    Returns:
        torch.Tensor: (..., T, 3, 3) TBN matrix for each face. Note TBN vectors are normalized but not necessarily orthognal
    """
    e01 = torch.index_select(pos, dim=-2, index=faces_pos[:, 1]) - torch.index_select(pos, dim=-2, index=faces_pos[:, 0])
    e02 = torch.index_select(pos, dim=-2, index=faces_pos[:, 2]) - torch.index_select(pos, dim=-2, index=faces_pos[:, 0])
    uv01 = torch.index_select(uv, dim=-2, index=faces_uv[:, 1]) - torch.index_select(uv, dim=-2, index=faces_uv[:, 0])
    uv02 = torch.index_select(uv, dim=-2, index=faces_uv[:, 2]) - torch.index_select(uv, dim=-2, index=faces_uv[:, 0])
    normal = torch.cross(e01, e02)
    tangent_bitangent = torch.stack([e01, e02], dim=-1) @ torch.inverse(torch.stack([uv01, uv02], dim=-1))
    tbn = torch.cat([tangent_bitangent, normal.unsqueeze(-1)], dim=-1)
    tbn = tbn / (torch.norm(tbn, p=2, dim=-2, keepdim=True) + eps)
    return tbn


def compute_vertex_tbn(faces_topo: torch.Tensor, pos: torch.Tensor, faces_pos: torch.Tensor, uv: torch.Tensor, faces_uv: torch.Tensor) -> torch.Tensor:
    """compute TBN matrix for each face

    Args:
        faces_topo (torch.Tensor): (T, 3), face indice of topology
        pos (torch.Tensor): shape (..., N_pos, 3), positions
        faces_pos (torch.Tensor): shape(T, 3) 
        uv (torch.Tensor): shape (..., N_uv, 3) uv coordinates, 
        faces_uv (torch.Tensor): shape(T, 3) 
        
    Returns:
        torch.Tensor: (..., V, 3, 3) TBN matrix for each face. Note TBN vectors are normalized but not necessarily orthognal
    """
    n_vertices = faces_topo.max().item() + 1
    n_tri = faces_topo.shape[-2]
    batch_shape = pos.shape[:-2]
    face_tbn = compute_face_tbn(pos, faces_pos, uv, faces_uv)    # (..., T, 3, 3)
    face_tbn = face_tbn[..., :, None, :, :].repeat(*[1] * len(batch_shape), 1, 3, 1, 1).view(*batch_shape, n_tri * 3, 3, 3)   # (..., T * 3, 3, 3)
    vertex_tbn = torch.index_add(torch.zeros(*batch_shape, n_vertices, 3, 3).to(face_tbn), dim=-3, index=faces_topo.view(-1), source=face_tbn)
    vertex_tbn = vertex_tbn / (torch.norm(vertex_tbn, p=2, dim=-2, keepdim=True) + 1e-7)
    return vertex_tbn


def laplacian(vertices: torch.Tensor, faces: torch.Tensor, weight: str = 'uniform') -> torch.Tensor:
    """Laplacian smooth with cotangent weights

    Args:
        vertices (torch.Tensor): shape (..., N, 3)
        faces (torch.Tensor): shape (T, 3)
        weight (str): 'uniform' or 'cotangent'
    """
    sum_verts = torch.zeros_like(vertices)                          # (..., N, 3)
    sum_weights = torch.zeros(*vertices.shape[:-1]).to(vertices)    # (..., N)
    face_verts = torch.index_select(vertices, -2, faces.view(-1)).view(*vertices.shape[:-2], *faces.shape, vertices.shape[-1])   # (..., T, 3)
    if weight == 'cotangent':
        for i in range(3):
            e1 = face_verts[..., (i + 1) % 3, :] - face_verts[..., i, :]
            e2 = face_verts[..., (i + 2) % 3, :] - face_verts[..., i, :]
            cot_angle = (e1 * e2).sum(dim=-1) / torch.cross(e1, e2, dim=-1).norm(p=2, dim=-1)   # (..., T, 3)
            sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 1) % 3], face_verts[..., (i + 2) % 3, :] * cot_angle[..., None])
            sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 1) % 3], cot_angle)
            sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 2) % 3], face_verts[..., (i + 1) % 3, :] * cot_angle[..., None])
            sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 2) % 3], cot_angle)
    elif weight == 'uniform':
        for i in range(3):
            sum_verts = torch.index_add(sum_verts, -2, faces[:, i], face_verts[..., (i + 1) % 3, :])
            sum_weights = torch.index_add(sum_weights, -1, faces[:, i], torch.ones_like(face_verts[..., i, 0]))
    else:
        raise NotImplementedError
    return sum_verts / (sum_weights[..., None] + 1e-7)


def laplacian_smooth_mesh(vertices: torch.Tensor, faces: torch.Tensor, weight: str = 'uniform', times: int = 5) -> torch.Tensor:
    """Laplacian smooth with cotangent weights

    Args:
        vertices (torch.Tensor): shape (..., N, 3)
        faces (torch.Tensor): shape (T, 3)
        weight (str): 'uniform' or 'cotangent'
    """
    for _ in range(times):
        vertices = laplacian(vertices, faces, weight)
    return vertices


def taubin_smooth_mesh(vertices: torch.Tensor, faces: torch.Tensor, lambda_: float = 0.5, mu_: float = -0.51) -> torch.Tensor:
    """Taubin smooth mesh

    Args:
        vertices (torch.Tensor): _description_
        faces (torch.Tensor): _description_
        lambda_ (float, optional): _description_. Defaults to 0.5.
        mu_ (float, optional): _description_. Defaults to -0.51.

    Returns:
        torch.Tensor: _description_
    """
    pt = vertices + lambda_ * laplacian_smooth_mesh(vertices, faces)
    p = pt + mu_ * laplacian_smooth_mesh(pt, faces)
    return p


def laplacian_hc_smooth_mesh(vertices: torch.Tensor, faces: torch.Tensor, times: int = 5, alpha: float = 0.5, beta: float = 0.5, weight: str = 'uniform'):
    """HC algorithm from Improved Laplacian Smoothing of Noisy Surface Meshes by J.Vollmer et al.
    """
    p = vertices
    for i in range(times):
        q = p
        p = laplacian_smooth_mesh(vertices, faces, weight)
        b = p - (alpha * vertices + (1 - alpha) * q)
        p = p - (beta * b + (1 - beta) * laplacian_smooth_mesh(b, faces, weight)) * 0.8
    return p
