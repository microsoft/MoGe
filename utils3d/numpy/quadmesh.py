import numpy as np
import scipy as sp
import scipy.optimize as spopt
from typing import *


__all__ = [
    'calc_quad_candidates',
    'calc_quad_distortion',
    'calc_quad_direction',
    'calc_quad_smoothness',
    'sovle_quad',
    'sovle_quad_qp',
    'tri_to_quad'
]


def calc_quad_candidates(
    edges: np.ndarray,
    face2edge: np.ndarray,
    edge2face: np.ndarray,
):
    """
    Calculate the candidate quad faces.

    Args:
        edges (np.ndarray): [E, 2] edge indices
        face2edge (np.ndarray): [T, 3] face to edge relation
        edge2face (np.ndarray): [E, 2] edge to face relation

    Returns:
        quads (np.ndarray): [Q, 4] quad candidate indices
        quad2edge (np.ndarray): [Q, 4] edge to quad candidate relation
        quad2adj (np.ndarray): [Q, 8] adjacent quad candidates of each quad candidate
        quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid
    """
    E = edges.shape[0]
    T = face2edge.shape[0]

    quads_valid = edge2face[:, 1] != -1
    Q = quads_valid.sum()
    quad2face = edge2face[quads_valid]  # [Q, 2]
    quad2edge = face2edge[quad2face]  # [Q, 2, 3]
    flag = quad2edge == np.arange(E)[quads_valid][:, None, None] # [Q, 2, 3]
    flag = flag.argmax(axis=-1)  # [Q, 2]
    quad2edge = np.stack([
        quad2edge[np.arange(Q)[:, None], np.arange(2)[None, :], (flag + 1) % 3],
        quad2edge[np.arange(Q)[:, None], np.arange(2)[None, :], (flag + 2) % 3],
    ], axis=-1).reshape(Q, 4)  # [Q, 4]

    quads = np.concatenate([
        np.where(
            (edges[quad2edge[:, 0:1], 1:] == edges[quad2edge[:, 1:2], :]).any(axis=-1),
            edges[quad2edge[:, 0:1], [[0, 1]]],
            edges[quad2edge[:, 0:1], [[1, 0]]],
        ),
        np.where(
            (edges[quad2edge[:, 2:3], 1:] == edges[quad2edge[:, 3:4], :]).any(axis=-1),
            edges[quad2edge[:, 2:3], [[0, 1]]],
            edges[quad2edge[:, 2:3], [[1, 0]]],
        ),
    ], axis=1)  # [Q, 4]

    quad2adj = edge2face[quad2edge]  # [Q, 4, 2]
    quad2adj = quad2adj[quad2adj != quad2face[:, [0,0,1,1], None]].reshape(Q, 4)  # [Q, 4]
    quad2adj_valid = quad2adj != -1
    quad2adj = face2edge[quad2adj]  # [Q, 4, 3]
    quad2adj[~quad2adj_valid, 0] = quad2edge[~quad2adj_valid]
    quad2adj[~quad2adj_valid, 1:] = -1
    quad2adj = quad2adj[quad2adj != quad2edge[..., None]].reshape(Q, 8)  # [Q, 8]
    edge_valid = -np.ones(E, dtype=np.int32)
    edge_valid[quads_valid] = np.arange(Q)
    quad2adj_valid = quad2adj != -1
    quad2adj[quad2adj_valid] = edge_valid[quad2adj[quad2adj_valid]]  # [Q, 8]

    return quads, quad2edge, quad2adj, quads_valid


def calc_quad_distortion(
    vertices: np.ndarray,
    quads: np.ndarray,
):
    """
    Calculate the distortion of each candidate quad face.

    Args:
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        quads (np.ndarray): [Q, 4] quad face indices

    Returns:
        distortion (np.ndarray): [Q] distortion of each quad face
    """
    edge0 = vertices[quads[:, 1]] - vertices[quads[:, 0]]  # [Q, 3]
    edge1 = vertices[quads[:, 2]] - vertices[quads[:, 1]]  # [Q, 3]
    edge2 = vertices[quads[:, 3]] - vertices[quads[:, 2]]  # [Q, 3]
    edge3 = vertices[quads[:, 0]] - vertices[quads[:, 3]]  # [Q, 3]
    cross = vertices[quads[:, 0]] - vertices[quads[:, 2]]  # [Q, 3]

    len0 = np.maximum(np.linalg.norm(edge0, axis=-1), 1e-10)  # [Q]
    len1 = np.maximum(np.linalg.norm(edge1, axis=-1), 1e-10)  # [Q]
    len2 = np.maximum(np.linalg.norm(edge2, axis=-1), 1e-10)  # [Q]
    len3 = np.maximum(np.linalg.norm(edge3, axis=-1), 1e-10)  # [Q]
    len_cross = np.maximum(np.linalg.norm(cross, axis=-1), 1e-10)  # [Q]

    angle0 = np.arccos(np.clip(np.sum(-edge0 * edge1, axis=-1) / (len0 * len1), -1, 1))  # [Q]
    angle1 = np.arccos(np.clip(np.sum(-edge1 * cross, axis=-1) / (len1 * len_cross), -1, 1)) \
           + np.arccos(np.clip(np.sum(cross * edge2, axis=-1) / (len_cross * len2), -1, 1))  # [Q]
    angle2 = np.arccos(np.clip(np.sum(-edge2 * edge3, axis=-1) / (len2 * len3), -1, 1))  # [Q]
    angle3 = np.arccos(np.clip(np.sum(-edge3 * -cross, axis=-1) / (len3 * len_cross), -1, 1)) \
           + np.arccos(np.clip(np.sum(-cross * edge0, axis=-1) / (len_cross * len0), -1, 1))  # [Q]

    normal0 = np.cross(edge0, edge1)  # [Q, 3]
    normal1 = np.cross(edge2, edge3)  # [Q, 3]
    normal0 = normal0 / np.maximum(np.linalg.norm(normal0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    normal1 = normal1 / np.maximum(np.linalg.norm(normal1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    angle_normal = np.arccos(np.clip(np.sum(normal0 * normal1, axis=-1), -1, 1))  # [Q]

    D90 = np.pi / 2
    D180 = np.pi
    D360 = np.pi * 2
    ang_eng = (np.abs(angle0 - D90)**2 + np.abs(angle1 - D90)**2 + np.abs(angle2 - D90)**2 + np.abs(angle3 - D90)**2) / 4  # [Q]
    dist_eng = np.abs(angle0 - angle2)**2 / np.minimum(np.maximum(np.minimum(angle0, angle2), 1e-10), np.maximum(D180 - np.maximum(angle0, angle2), 1e-10)) \
             + np.abs(angle1 - angle3)**2 / np.minimum(np.maximum(np.minimum(angle1, angle3), 1e-10), np.maximum(D180 - np.maximum(angle1, angle3), 1e-10))  # [Q]
    plane_eng = np.where(angle_normal < D90/2, np.abs(angle_normal)**2, 1e10)  # [Q]
    eng = ang_eng + 2 * dist_eng + 2 * plane_eng  # [Q]

    return eng


def calc_quad_direction(
        vertices: np.ndarray,
        quads: np.ndarray,
    ):
    """
    Calculate the direction of each candidate quad face.

    Args:
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        quads (np.ndarray): [Q, 4] quad face indices

    Returns:
        direction (np.ndarray): [Q, 4] direction of each quad face.
            Represented by the angle between the crossing and each edge.
    """
    mid0 = (vertices[quads[:, 0]] + vertices[quads[:, 1]]) / 2  # [Q, 3]
    mid1 = (vertices[quads[:, 1]] + vertices[quads[:, 2]]) / 2  # [Q, 3]
    mid2 = (vertices[quads[:, 2]] + vertices[quads[:, 3]]) / 2  # [Q, 3]
    mid3 = (vertices[quads[:, 3]] + vertices[quads[:, 0]]) / 2  # [Q, 3]

    cross0 = mid2 - mid0  # [Q, 3]
    cross1 = mid3 - mid1  # [Q, 3]
    cross0 = cross0 / np.maximum(np.linalg.norm(cross0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    cross1 = cross1 / np.maximum(np.linalg.norm(cross1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]

    edge0 = vertices[quads[:, 1]] - vertices[quads[:, 0]]  # [Q, 3]
    edge1 = vertices[quads[:, 2]] - vertices[quads[:, 1]]  # [Q, 3]
    edge2 = vertices[quads[:, 3]] - vertices[quads[:, 2]]  # [Q, 3]
    edge3 = vertices[quads[:, 0]] - vertices[quads[:, 3]]  # [Q, 3]
    edge0 = edge0 / np.maximum(np.linalg.norm(edge0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    edge1 = edge1 / np.maximum(np.linalg.norm(edge1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    edge2 = edge2 / np.maximum(np.linalg.norm(edge2, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    edge3 = edge3 / np.maximum(np.linalg.norm(edge3, axis=-1, keepdims=True), 1e-10)  # [Q, 3]

    direction = np.stack([
        np.arccos(np.clip(np.sum(cross0 * edge0, axis=-1), -1, 1)),
        np.arccos(np.clip(np.sum(cross1 * edge1, axis=-1), -1, 1)),
        np.arccos(np.clip(np.sum(-cross0 * edge2, axis=-1), -1, 1)),
        np.arccos(np.clip(np.sum(-cross1 * edge3, axis=-1), -1, 1)),
    ], axis=-1)  # [Q, 4]

    return direction


def calc_quad_smoothness(
        quad2edge: np.ndarray,
        quad2adj: np.ndarray,
        quads_direction: np.ndarray,
    ):
    """
    Calculate the smoothness of each candidate quad face connection.

    Args:
        quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
        quads_direction (np.ndarray): [Q, 4] direction of each quad face

    Returns:
        smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
    """
    Q = quad2adj.shape[0]
    quad2adj_valid = quad2adj != -1
    connections = np.stack([
        np.arange(Q)[:, None].repeat(8, axis=1),
        quad2adj,
    ], axis=-1)[quad2adj_valid]  # [C, 2]
    shared_edge_idx_0 = np.array([[0, 0, 1, 1, 2, 2, 3, 3]]).repeat(Q, axis=0)[quad2adj_valid]  # [C]
    shared_edge_idx_1 = np.argmax(quad2edge[quad2adj][quad2adj_valid] == quad2edge[connections[:, 0], shared_edge_idx_0][:, None], axis=-1)  # [C]
    valid_smoothness = np.abs(quads_direction[connections[:, 0], shared_edge_idx_0] - quads_direction[connections[:, 1], shared_edge_idx_1])**2  # [C]
    smoothness = np.zeros([Q, 8], dtype=np.float32)
    smoothness[quad2adj_valid] = valid_smoothness
    return smoothness


def sovle_quad(
        face2edge: np.ndarray,
        edge2face: np.ndarray,
        quad2adj: np.ndarray,
        quads_distortion: np.ndarray,
        quads_smoothness: np.ndarray,
        quads_valid: np.ndarray,
    ):
    """
    Solve the quad mesh from the candidate quad faces.

    Args:
        face2edge (np.ndarray): [T, 3] face to edge relation
        edge2face (np.ndarray): [E, 2] edge to face relation
        quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
        quads_distortion (np.ndarray): [Q] distortion of each quad face
        quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
        quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

    Returns:
        weights (np.ndarray): [Q] weight of each valid quad face
    """
    T = face2edge.shape[0]
    E = edge2face.shape[0]
    Q = quads_distortion.shape[0]
    edge_valid = -np.ones(E, dtype=np.int32)
    edge_valid[quads_valid] = np.arange(Q)

    quads_connection = np.stack([
        np.arange(Q)[:, None].repeat(8, axis=1),
        quad2adj,
    ], axis=-1)[quad2adj != -1]  # [C, 2]
    quads_connection = np.sort(quads_connection, axis=-1)  # [C, 2]
    quads_connection, quads_connection_idx = np.unique(quads_connection, axis=0, return_index=True)  # [C, 2], [C]
    quads_smoothness = quads_smoothness[quad2adj != -1]  # [C]
    quads_smoothness = quads_smoothness[quads_connection_idx]  # [C]
    C = quads_connection.shape[0]

    # Construct the linear programming problem

    # Variables:
    #   quads_weight: [Q] weight of each quad face
    #   tri_min_weight: [T] minimum weight of each triangle face
    #   conn_min_weight: [C] minimum weight of each quad face connection
    #   conn_max_weight: [C] maximum weight of each quad face connection
    # Objective:
    #   mimi

    c = np.concatenate([
        quads_distortion - 3,
        quads_smoothness*4 - 2,
        quads_smoothness*4,
    ], axis=0)  # [Q+C]

    A_ub_triplet = np.concatenate([
        np.stack([np.arange(T), edge_valid[face2edge[:, 0]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 1]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 2]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T, T+C), np.arange(Q, Q+C), np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T, T+C), quads_connection[:, 0], -np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T, T+C), quads_connection[:, 1], -np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T+C, T+2*C), np.arange(Q+C, Q+2*C), -np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T+C, T+2*C), quads_connection[:, 0], np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T+C, T+2*C), quads_connection[:, 1], np.ones(C)], axis=1),  # [C, 3]
    ], axis=0)  # [3T+6C, 3]
    A_ub_triplet = A_ub_triplet[A_ub_triplet[:, 1] != -1]  # [3T', 3]
    A_ub = sp.sparse.coo_matrix((A_ub_triplet[:, 2], (A_ub_triplet[:, 0], A_ub_triplet[:, 1])), shape=[T+2*C, Q+2*C])  # [T, 
    b_ub = np.concatenate([np.ones(T), -np.ones(C), np.ones(C)], axis=0)  # [T+2C]
    bound = np.stack([
        np.concatenate([np.zeros(Q), -np.ones(C), np.zeros(C)], axis=0),
        np.concatenate([np.ones(Q), np.ones(C), np.ones(C)], axis=0),
    ], axis=1)  # [Q+2C, 2]
    A_eq = None
    b_eq = None

    print('Solver statistics:')
    print(f'    #T = {T}')
    print(f'    #Q = {Q}')
    print(f'    #C = {C}')

    # Solve the linear programming problem
    last_num_valid = 0
    for i in range(100):
        res_ = spopt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bound)
        if not res_.success:
            print(f'    Iter {i} | Failed with {res_.message}')
            break
        res = res_
        weights = res.x[:Q]
        valid = (weights > 0.5)
        num_valid = valid.sum()
        print(f'    Iter {i} | #Q_valid = {num_valid}')
        if num_valid == last_num_valid:
            break
        last_num_valid = num_valid
        A_eq_triplet = np.stack([
            np.arange(num_valid),
            np.arange(Q)[valid],
            np.ones(num_valid),
        ], axis=1)  # [num_valid, 3]
        A_eq = sp.sparse.coo_matrix((A_eq_triplet[:, 2], (A_eq_triplet[:, 0], A_eq_triplet[:, 1])), shape=[num_valid, Q+2*C])  # [num_valid, Q+C]
        b_eq = np.where(weights[valid] > 0.5, 1, 0)  # [num_valid]

    # Return the result
    quads_weight = res.x[:Q]
    conn_min_weight = res.x[Q:Q+C]
    conn_max_weight = res.x[Q+C:Q+2*C]
    return quads_weight, conn_min_weight, conn_max_weight


def sovle_quad_qp(
        face2edge: np.ndarray,
        edge2face: np.ndarray,
        quad2adj: np.ndarray,
        quads_distortion: np.ndarray,
        quads_smoothness: np.ndarray,
        quads_valid: np.ndarray,
    ):
    """
    Solve the quad mesh from the candidate quad faces.

    Args:
        face2edge (np.ndarray): [T, 3] face to edge relation
        edge2face (np.ndarray): [E, 2] edge to face relation
        quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
        quads_distortion (np.ndarray): [Q] distortion of each quad face
        quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
        quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

    Returns:
        weights (np.ndarray): [Q] weight of each valid quad face
    """
    T = face2edge.shape[0]
    E = edge2face.shape[0]
    Q = quads_distortion.shape[0]
    edge_valid = -np.ones(E, dtype=np.int32)
    edge_valid[quads_valid] = np.arange(Q)

    # Construct the quadratic programming problem
    C_smoothness_triplet = np.stack([
        np.arange(Q)[:, None].repeat(8, axis=1)[quad2adj != -1],
        quad2adj[quad2adj != -1],
        5 * quads_smoothness[quad2adj != -1],
    ], axis=-1)  # [C, 3]
    # C_smoothness_triplet = np.concatenate([
    #     C_smoothness_triplet,
    #     np.stack([np.arange(Q), np.arange(Q), 20*np.ones(Q)], axis=1),
    # ], axis=0)  # [C+Q, 3]
    C_smoothness = sp.sparse.coo_matrix((C_smoothness_triplet[:, 2], (C_smoothness_triplet[:, 0], C_smoothness_triplet[:, 1])), shape=[Q, Q])  # [Q, Q]
    C_smoothness = C_smoothness.tocsc()
    C_dist = quads_distortion - 20  # [Q]

    A_eq = sp.sparse.coo_matrix((np.zeros(Q), (np.zeros(Q), np.arange(Q))), shape=[1, Q])  # [1, Q]\
    A_eq = A_eq.tocsc()
    b_eq = np.array([0])

    A_ub_triplet = np.concatenate([
        np.stack([np.arange(T), edge_valid[face2edge[:, 0]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 1]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 2]], np.ones(T)], axis=1),  # [T, 3]
    ], axis=0)  # [3T, 3]
    A_ub_triplet = A_ub_triplet[A_ub_triplet[:, 1] != -1]  # [3T', 3]
    A_ub = sp.sparse.coo_matrix((A_ub_triplet[:, 2], (A_ub_triplet[:, 0], A_ub_triplet[:, 1])), shape=[T, Q])  # [T, Q]
    A_ub = A_ub.tocsc()
    b_ub = np.ones(T)

    lb = np.zeros(Q)
    ub = np.ones(Q)

    import piqp
    solver = piqp.SparseSolver()
    solver.settings.verbose = True
    solver.settings.compute_timings = True
    solver.setup(C_smoothness, C_dist, A_eq, b_eq, A_ub, b_ub, lb, ub)

    status = solver.solve()

    # x = cp.Variable(Q)
    # prob = cp.Problem(
    #     cp.Minimize(cp.quad_form(x, C_smoothness) + C_dist.T @ x),
    #     [
    #         A_ub @ x <= b_ub,
    #         x >= 0, x <= 1,
    #     ]
    # )

    # # Solve the quadratic programming problem
    # prob.solve(solver=cp.PIQP, verbose=True)

    # Return the result
    weights = solver.result.x
    return weights


def tri_to_quad(
        vertices: np.ndarray,
        faces: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a triangle mesh to a quad mesh.
    NOTE: The input mesh must be a manifold mesh.

    Args:
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices

    Returns:
        vertices (np.ndarray): [N_, 3] 3-dimensional vertices
        faces (np.ndarray): [Q, 4] quad face indices
    """
    raise NotImplementedError


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    import utils3d
    import numpy as np
    import cv2
    from vis import vis_edge_color

    file = 'miku'

    vertices, faces = utils3d.io.read_ply(f'test/assets/{file}.ply')
    edges, edge2face, face2edge, face2face = calc_relations(faces)
    quad_cands, quad2edge, quad2adj, quad_valid = calc_quad_candidates(edges, face2edge, edge2face)
    distortion = calc_quad_distortion(vertices, quad_cands)
    direction = calc_quad_direction(vertices, quad_cands)
    smoothness = calc_quad_smoothness(quad2edge, quad2adj, direction)
    boundary_edges = edges[edge2face[:, 1] == -1]
    quads_weight, conn_min_weight, conn_max_weight = sovle_quad(face2edge, edge2face, quad2adj, distortion, smoothness, quad_valid)
    quads = quad_cands[quads_weight > 0.5]
    print('Mesh statistics')
    print(f'    #V      =   {vertices.shape[0]}')
    print(f'    #F      =   {faces.shape[0]}')
    print(f'    #E      =   {edges.shape[0]}')
    print(f'    #B      =   {boundary_edges.shape[0]}')
    print(f'    #Q_cand =   {quad_cands.shape[0]}')
    print(f'    #Q      =   {quads.shape[0]}')

    utils3d.io.write_ply(f'test/assets/{file}_boundary_edges.ply', vertices=vertices, edges=boundary_edges)
    utils3d.io.write_ply(f'test/assets/{file}_quad_candidates.ply', vertices=vertices, faces=quads)

    edge_colors = np.zeros([edges.shape[0], 3], dtype=np.uint8)
    distortion = (distortion - distortion.min()) / (distortion.max() - distortion.min())
    distortion = (distortion * 255).astype(np.uint8)
    edge_colors[quad_valid] = cv2.cvtColor(cv2.applyColorMap(distortion, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    utils3d.io.write_ply(f'test/assets/{file}_quad_candidates_distortion.ply', **vis_edge_color(vertices, edges, edge_colors))

    edge_colors = np.zeros([edges.shape[0], 3], dtype=np.uint8)
    edge_colors[quad_valid] = cv2.cvtColor(cv2.applyColorMap((quads_weight * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    utils3d.io.write_ply(f'test/assets/{file}_quad_candidates_weights.ply', **vis_edge_color(vertices, edges, edge_colors))
    utils3d.io.write_ply(f'test/assets/{file}_quad.ply', vertices=vertices, faces=quads)

    quad_centers = vertices[quad_cands].mean(axis=1)
    conns = np.stack([
        np.arange(quad_cands.shape[0])[:, None].repeat(8, axis=1),
        quad2adj,
    ], axis=-1)[quad2adj != -1]  # [C, 2]
    conns, conns_idx = np.unique(np.sort(conns, axis=-1), axis=0, return_index=True)  # [C, 2], [C]
    smoothness = smoothness[quad2adj != -1][conns_idx]  # [C]
    conns_color = cv2.cvtColor(cv2.applyColorMap((smoothness * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    utils3d.io.write_ply(f'test/assets/{file}_quad_conn_smoothness.ply', **vis_edge_color(quad_centers, conns, conns_color))
    conns_color = cv2.cvtColor(cv2.applyColorMap((conn_min_weight * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    utils3d.io.write_ply(f'test/assets/{file}_quad_conn_min.ply', **vis_edge_color(quad_centers, conns, conns_color))
    conns_color = cv2.cvtColor(cv2.applyColorMap((conn_max_weight * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    utils3d.io.write_ply(f'test/assets/{file}_quad_conn_max.ply', **vis_edge_color(quad_centers, conns, conns_color))
    
    