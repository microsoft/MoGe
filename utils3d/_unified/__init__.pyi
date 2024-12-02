# Auto-generated interface file
from typing import List, Tuple, Dict, Union, Optional, Any, overload, Literal, Callable
import numpy as numpy_
import torch as torch_
import nvdiffrast.torch
import numbers
from . import numpy, torch
import utils3d.numpy, utils3d.torch

__all__ = ["triangulate", 
"compute_face_normal", 
"compute_face_angle", 
"compute_vertex_normal", 
"compute_vertex_normal_weighted", 
"remove_corrupted_faces", 
"merge_duplicate_vertices", 
"remove_unreferenced_vertices", 
"subdivide_mesh_simple", 
"mesh_relations", 
"flatten_mesh_indices", 
"calc_quad_candidates", 
"calc_quad_distortion", 
"calc_quad_direction", 
"calc_quad_smoothness", 
"sovle_quad", 
"sovle_quad_qp", 
"tri_to_quad", 
"sliding_window_1d", 
"sliding_window_nd", 
"sliding_window_2d", 
"max_pool_1d", 
"max_pool_2d", 
"max_pool_nd", 
"depth_edge", 
"normals_edge", 
"depth_aliasing", 
"interpolate", 
"image_scrcoord", 
"image_uv", 
"image_pixel_center", 
"image_pixel", 
"image_mesh", 
"image_mesh_from_depth", 
"depth_to_normals", 
"points_to_normals", 
"chessboard", 
"cube", 
"icosahedron", 
"square", 
"camera_frustum", 
"perspective", 
"perspective_from_fov", 
"perspective_from_fov_xy", 
"intrinsics_from_focal_center", 
"intrinsics_from_fov", 
"fov_to_focal", 
"focal_to_fov", 
"intrinsics_to_fov", 
"view_look_at", 
"extrinsics_look_at", 
"perspective_to_intrinsics", 
"perspective_to_near_far", 
"intrinsics_to_perspective", 
"extrinsics_to_view", 
"view_to_extrinsics", 
"normalize_intrinsics", 
"crop_intrinsics", 
"pixel_to_uv", 
"pixel_to_ndc", 
"uv_to_pixel", 
"project_depth", 
"depth_buffer_to_linear", 
"unproject_cv", 
"unproject_gl", 
"project_cv", 
"project_gl", 
"quaternion_to_matrix", 
"axis_angle_to_matrix", 
"matrix_to_quaternion", 
"extrinsics_to_essential", 
"euler_axis_angle_rotation", 
"euler_angles_to_matrix", 
"skew_symmetric", 
"rotation_matrix_from_vectors", 
"ray_intersection", 
"se3_matrix", 
"slerp_quaternion", 
"slerp_vector", 
"lerp", 
"lerp_se3_matrix", 
"piecewise_lerp", 
"piecewise_lerp_se3_matrix", 
"apply_transform", 
"linear_spline_interpolate", 
"RastContext", 
"rasterize_triangle_faces", 
"rasterize_edges", 
"texture", 
"warp_image_by_depth", 
"test_rasterization", 
"compute_face_angles", 
"compute_face_tbn", 
"compute_vertex_tbn", 
"laplacian", 
"laplacian_smooth_mesh", 
"taubin_smooth_mesh", 
"laplacian_hc_smooth_mesh", 
"get_rays", 
"get_image_rays", 
"get_mipnerf_cones", 
"volume_rendering", 
"bin_sample", 
"importance_sample", 
"nerf_render_rays", 
"mipnerf_render_rays", 
"nerf_render_view", 
"mipnerf_render_view", 
"InstantNGP", 
"point_to_normal", 
"depth_to_normal", 
"masked_min", 
"masked_max", 
"bounding_rect", 
"intrinsics_from_fov_xy", 
"matrix_to_euler_angles", 
"matrix_to_axis_angle", 
"axis_angle_to_quaternion", 
"quaternion_to_axis_angle", 
"slerp", 
"interpolate_extrinsics", 
"interpolate_view", 
"to4x4", 
"rotation_matrix_2d", 
"rotate_2d", 
"translate_2d", 
"scale_2d", 
"apply_2d", 
"warp_image_by_forward_flow"]

@overload
def triangulate(faces: numpy_.ndarray, vertices: numpy_.ndarray = None, backslash: numpy_.ndarray = None) -> numpy_.ndarray:
    """Triangulate a polygonal mesh.

Args:
    faces (np.ndarray): [L, P] polygonal faces
    vertices (np.ndarray, optional): [N, 3] 3-dimensional vertices.
        If given, the triangulation is performed according to the distance
        between vertices. Defaults to None.
    backslash (np.ndarray, optional): [L] boolean array indicating
        how to triangulate the quad faces. Defaults to None.

Returns:
    (np.ndarray): [L * (P - 2), 3] triangular faces"""
    utils3d.numpy.mesh.triangulate

@overload
def compute_face_normal(vertices: numpy_.ndarray, faces: numpy_.ndarray) -> numpy_.ndarray:
    """Compute face normals of a triangular mesh

Args:
    vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices

Returns:
    normals (np.ndarray): [..., T, 3] face normals"""
    utils3d.numpy.mesh.compute_face_normal

@overload
def compute_face_angle(vertices: numpy_.ndarray, faces: numpy_.ndarray, eps: float = 1e-12) -> numpy_.ndarray:
    """Compute face angles of a triangular mesh

Args:
    vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices

Returns:
    angles (np.ndarray): [..., T, 3] face angles"""
    utils3d.numpy.mesh.compute_face_angle

@overload
def compute_vertex_normal(vertices: numpy_.ndarray, faces: numpy_.ndarray, face_normal: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute vertex normals of a triangular mesh by averaging neightboring face normals
TODO: can be improved.

Args:
    vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices
    face_normal (np.ndarray, optional): [..., T, 3] face normals.
        None to compute face normals from vertices and faces. Defaults to None.

Returns:
    normals (np.ndarray): [..., N, 3] vertex normals"""
    utils3d.numpy.mesh.compute_vertex_normal

@overload
def compute_vertex_normal_weighted(vertices: numpy_.ndarray, faces: numpy_.ndarray, face_normal: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute vertex normals of a triangular mesh by weighted sum of neightboring face normals
according to the angles

Args:
    vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
    faces (np.ndarray): [..., T, 3] triangular face indices
    face_normal (np.ndarray, optional): [..., T, 3] face normals.
        None to compute face normals from vertices and faces. Defaults to None.

Returns:
    normals (np.ndarray): [..., N, 3] vertex normals"""
    utils3d.numpy.mesh.compute_vertex_normal_weighted

@overload
def remove_corrupted_faces(faces: numpy_.ndarray) -> numpy_.ndarray:
    """Remove corrupted faces (faces with duplicated vertices)

Args:
    faces (np.ndarray): [T, 3] triangular face indices

Returns:
    np.ndarray: [T_, 3] triangular face indices"""
    utils3d.numpy.mesh.remove_corrupted_faces

@overload
def merge_duplicate_vertices(vertices: numpy_.ndarray, faces: numpy_.ndarray, tol: float = 1e-06) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Merge duplicate vertices of a triangular mesh. 
Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

Args:
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices
    tol (float, optional): tolerance for merging. Defaults to 1e-6.

Returns:
    vertices (np.ndarray): [N_, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices"""
    utils3d.numpy.mesh.merge_duplicate_vertices

@overload
def remove_unreferenced_vertices(faces: numpy_.ndarray, *vertice_attrs, return_indices: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Remove unreferenced vertices of a mesh. 
Unreferenced vertices are removed, and the face indices are updated accordingly.

Args:
    faces (np.ndarray): [T, P] face indices
    *vertice_attrs: vertex attributes

Returns:
    faces (np.ndarray): [T, P] face indices
    *vertice_attrs: vertex attributes
    indices (np.ndarray, optional): [N] indices of vertices that are kept. Defaults to None."""
    utils3d.numpy.mesh.remove_unreferenced_vertices

@overload
def subdivide_mesh_simple(vertices: numpy_.ndarray, faces: numpy_.ndarray, n: int = 1) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.

Args:
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices
    n (int, optional): number of subdivisions. Defaults to 1.

Returns:
    vertices (np.ndarray): [N_, 3] subdivided 3-dimensional vertices
    faces (np.ndarray): [4 * T, 3] subdivided triangular face indices"""
    utils3d.numpy.mesh.subdivide_mesh_simple

@overload
def mesh_relations(faces: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Calculate the relation between vertices and faces.
NOTE: The input mesh must be a manifold triangle mesh.

Args:
    faces (np.ndarray): [T, 3] triangular face indices

Returns:
    edges (np.ndarray): [E, 2] edge indices
    edge2face (np.ndarray): [E, 2] edge to face relation. The second column is -1 if the edge is boundary.
    face2edge (np.ndarray): [T, 3] face to edge relation
    face2face (np.ndarray): [T, 3] face to face relation"""
    utils3d.numpy.mesh.mesh_relations

@overload
def flatten_mesh_indices(*args: numpy_.ndarray) -> Tuple[numpy_.ndarray, ...]:
    utils3d.numpy.mesh.flatten_mesh_indices

@overload
def calc_quad_candidates(edges: numpy_.ndarray, face2edge: numpy_.ndarray, edge2face: numpy_.ndarray):
    """Calculate the candidate quad faces.

Args:
    edges (np.ndarray): [E, 2] edge indices
    face2edge (np.ndarray): [T, 3] face to edge relation
    edge2face (np.ndarray): [E, 2] edge to face relation

Returns:
    quads (np.ndarray): [Q, 4] quad candidate indices
    quad2edge (np.ndarray): [Q, 4] edge to quad candidate relation
    quad2adj (np.ndarray): [Q, 8] adjacent quad candidates of each quad candidate
    quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid"""
    utils3d.numpy.quadmesh.calc_quad_candidates

@overload
def calc_quad_distortion(vertices: numpy_.ndarray, quads: numpy_.ndarray):
    """Calculate the distortion of each candidate quad face.

Args:
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    quads (np.ndarray): [Q, 4] quad face indices

Returns:
    distortion (np.ndarray): [Q] distortion of each quad face"""
    utils3d.numpy.quadmesh.calc_quad_distortion

@overload
def calc_quad_direction(vertices: numpy_.ndarray, quads: numpy_.ndarray):
    """Calculate the direction of each candidate quad face.

Args:
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    quads (np.ndarray): [Q, 4] quad face indices

Returns:
    direction (np.ndarray): [Q, 4] direction of each quad face.
        Represented by the angle between the crossing and each edge."""
    utils3d.numpy.quadmesh.calc_quad_direction

@overload
def calc_quad_smoothness(quad2edge: numpy_.ndarray, quad2adj: numpy_.ndarray, quads_direction: numpy_.ndarray):
    """Calculate the smoothness of each candidate quad face connection.

Args:
    quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
    quads_direction (np.ndarray): [Q, 4] direction of each quad face

Returns:
    smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection"""
    utils3d.numpy.quadmesh.calc_quad_smoothness

@overload
def sovle_quad(face2edge: numpy_.ndarray, edge2face: numpy_.ndarray, quad2adj: numpy_.ndarray, quads_distortion: numpy_.ndarray, quads_smoothness: numpy_.ndarray, quads_valid: numpy_.ndarray):
    """Solve the quad mesh from the candidate quad faces.

Args:
    face2edge (np.ndarray): [T, 3] face to edge relation
    edge2face (np.ndarray): [E, 2] edge to face relation
    quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
    quads_distortion (np.ndarray): [Q] distortion of each quad face
    quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
    quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

Returns:
    weights (np.ndarray): [Q] weight of each valid quad face"""
    utils3d.numpy.quadmesh.sovle_quad

@overload
def sovle_quad_qp(face2edge: numpy_.ndarray, edge2face: numpy_.ndarray, quad2adj: numpy_.ndarray, quads_distortion: numpy_.ndarray, quads_smoothness: numpy_.ndarray, quads_valid: numpy_.ndarray):
    """Solve the quad mesh from the candidate quad faces.

Args:
    face2edge (np.ndarray): [T, 3] face to edge relation
    edge2face (np.ndarray): [E, 2] edge to face relation
    quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
    quads_distortion (np.ndarray): [Q] distortion of each quad face
    quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
    quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

Returns:
    weights (np.ndarray): [Q] weight of each valid quad face"""
    utils3d.numpy.quadmesh.sovle_quad_qp

@overload
def tri_to_quad(vertices: numpy_.ndarray, faces: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Convert a triangle mesh to a quad mesh.
NOTE: The input mesh must be a manifold mesh.

Args:
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices

Returns:
    vertices (np.ndarray): [N_, 3] 3-dimensional vertices
    faces (np.ndarray): [Q, 4] quad face indices"""
    utils3d.numpy.quadmesh.tri_to_quad

@overload
def sliding_window_1d(x: numpy_.ndarray, window_size: int, stride: int, axis: int = -1):
    """Return x view of the input array with x sliding window of the given kernel size and stride.
The sliding window is performed over the given axis, and the window dimension is append to the end of the output array's shape.

Args:
    x (np.ndarray): input array with shape (..., axis_size, ...)
    kernel_size (int): size of the sliding window
    stride (int): stride of the sliding window
    axis (int): axis to perform sliding window over

Returns:
    a_sliding (np.ndarray): view of the input array with shape (..., n_windows, ..., kernel_size), where n_windows = (axis_size - kernel_size + 1) // stride"""
    utils3d.numpy.utils.sliding_window_1d

@overload
def sliding_window_nd(x: numpy_.ndarray, window_size: Tuple[int, ...], stride: Tuple[int, ...], axis: Tuple[int, ...]) -> numpy_.ndarray:
    utils3d.numpy.utils.sliding_window_nd

@overload
def sliding_window_2d(x: numpy_.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> numpy_.ndarray:
    utils3d.numpy.utils.sliding_window_2d

@overload
def max_pool_1d(x: numpy_.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    utils3d.numpy.utils.max_pool_1d

@overload
def max_pool_2d(x: numpy_.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    utils3d.numpy.utils.max_pool_2d

@overload
def max_pool_nd(x: numpy_.ndarray, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], axis: Tuple[int, ...]) -> numpy_.ndarray:
    utils3d.numpy.utils.max_pool_nd

@overload
def depth_edge(depth: numpy_.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.

Args:
    depth (np.ndarray): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

Returns:
    edge (np.ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.utils.depth_edge

@overload
def normals_edge(normals: numpy_.ndarray, tol: float, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the edge mask from normal map.

Args:
    normal (np.ndarray): shape (..., height, width, 3), normal map
    tol (float): tolerance in degrees

Returns:
    edge (np.ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.utils.normals_edge

@overload
def depth_aliasing(depth: numpy_.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the map that indicates the aliasing of x depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
Args:
    depth (np.ndarray): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

Returns:
    edge (np.ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.utils.depth_aliasing

@overload
def interpolate(bary: numpy_.ndarray, tri_id: numpy_.ndarray, attr: numpy_.ndarray, faces: numpy_.ndarray) -> numpy_.ndarray:
    """Interpolate with given barycentric coordinates and triangle indices

Args:
    bary (np.ndarray): shape (..., 3), barycentric coordinates
    tri_id (np.ndarray): int array of shape (...), triangle indices
    attr (np.ndarray): shape (N, M), vertices attributes
    faces (np.ndarray): int array of shape (T, 3), face vertex indices

Returns:
    np.ndarray: shape (..., M) interpolated result"""
    utils3d.numpy.utils.interpolate

@overload
def image_scrcoord(width: int, height: int) -> numpy_.ndarray:
    """Get OpenGL's screen space coordinates, ranging in [0, 1].
[0, 0] is the bottom-left corner of the image.

Args:
    width (int): image width
    height (int): image height

Returns:
    (np.ndarray): shape (height, width, 2)"""
    utils3d.numpy.utils.image_scrcoord

@overload
def image_uv(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get image space UV grid, ranging in [0, 1]. 

>>> image_uv(10, 10):
[[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
 [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
  ...             ...                  ...
 [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

Args:
    width (int): image width
    height (int): image height

Returns:
    np.ndarray: shape (height, width, 2)"""
    utils3d.numpy.utils.image_uv

@overload
def image_pixel_center(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get image pixel center coordinates, ranging in [0, width] and [0, height].
`image[i, j]` has pixel center coordinates `(j + 0.5, i + 0.5)`.

>>> image_pixel_center(10, 10):
[[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
 [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
  ...             ...                  ...
[[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]

Args:
    width (int): image width
    height (int): image height

Returns:
    np.ndarray: shape (height, width, 2)"""
    utils3d.numpy.utils.image_pixel_center

@overload
def image_pixel(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, dtype: numpy_.dtype = numpy_.int32) -> numpy_.ndarray:
    """Get image pixel coordinates grid, ranging in [0, width - 1] and [0, height - 1].
`image[i, j]` has pixel center coordinates `(j, i)`.

>>> image_pixel_center(10, 10):
[[[0, 0], [1, 0], ..., [9, 0]],
 [[0, 1.5], [1, 1], ..., [9, 1]],
  ...             ...                  ...
[[0, 9.5], [1, 9], ..., [9, 9 ]]]

Args:
    width (int): image width
    height (int): image height

Returns:
    np.ndarray: shape (height, width, 2)"""
    utils3d.numpy.utils.image_pixel

@overload
def image_mesh(*image_attrs: numpy_.ndarray, mask: numpy_.ndarray = None, tri: bool = False, return_indices: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

Args:
    *image_attrs (np.ndarray): image attributes in shape (height, width, [channels])
    mask (np.ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

Returns:
    faces (np.ndarray): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
    *vertex_attrs (np.ndarray): vertex attributes in corresponding order with input image_attrs
    indices (np.ndarray, optional): indices of vertices in the original mesh"""
    utils3d.numpy.utils.image_mesh

@overload
def image_mesh_from_depth(depth: numpy_.ndarray, extrinsics: numpy_.ndarray = None, intrinsics: numpy_.ndarray = None, *vertice_attrs: numpy_.ndarray, atol: float = None, rtol: float = None, remove_by_depth: bool = False, return_uv: bool = False, return_indices: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Get x triangle mesh by lifting depth map to 3D.

Args:
    depth (np.ndarray): [H, W] depth map
    extrinsics (np.ndarray, optional): [4, 4] extrinsics matrix. Defaults to None.
    intrinsics (np.ndarray, optional): [3, 3] intrinsics matrix. Defaults to None.
    *vertice_attrs (np.ndarray): [H, W, C] vertex attributes. Defaults to None.
    atol (float, optional): absolute tolerance. Defaults to None.
    rtol (float, optional): relative tolerance. Defaults to None.
        triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
    remove_by_depth (bool, optional): whether to remove triangles with large depth difference. Defaults to True.
    return_uv (bool, optional): whether to return uv coordinates. Defaults to False.
    return_indices (bool, optional): whether to return indices of vertices in the original mesh. Defaults to False.

Returns:
    vertices (np.ndarray): [N, 3] vertices
    faces (np.ndarray): [T, 3] faces
    *vertice_attrs (np.ndarray): [N, C] vertex attributes
    image_uv (np.ndarray, optional): [N, 2] uv coordinates
    ref_indices (np.ndarray, optional): [N] indices of vertices in the original mesh"""
    utils3d.numpy.utils.image_mesh_from_depth

@overload
def depth_to_normals(depth: numpy_.ndarray, intrinsics: numpy_.ndarray, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

Args:
    depth (np.ndarray): shape (height, width), linear depth map
    intrinsics (np.ndarray): shape (3, 3), intrinsics matrix
Returns:
    normal (np.ndarray): shape (height, width, 3), normal map. """
    utils3d.numpy.utils.depth_to_normals

@overload
def points_to_normals(point: numpy_.ndarray, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Calculate normal map from point map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

Args:
    point (np.ndarray): shape (height, width, 3), point map
Returns:
    normal (np.ndarray): shape (height, width, 3), normal map. """
    utils3d.numpy.utils.points_to_normals

@overload
def chessboard(width: int, height: int, grid_size: int, color_a: numpy_.ndarray, color_b: numpy_.ndarray) -> numpy_.ndarray:
    """get x chessboard image

Args:
    width (int): image width
    height (int): image height
    grid_size (int): size of chessboard grid
    color_a (np.ndarray): color of the grid at the top-left corner
    color_b (np.ndarray): color in complementary grid cells

Returns:
    image (np.ndarray): shape (height, width, channels), chessboard image"""
    utils3d.numpy.utils.chessboard

@overload
def cube(tri: bool = False) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Get x cube mesh of size 1 centered at origin.

### Parameters
    tri (bool, optional): return triangulated mesh. Defaults to False, which returns quad mesh.

### Returns
    vertices (np.ndarray): shape (8, 3) 
    faces (np.ndarray): shape (12, 3)"""
    utils3d.numpy.utils.cube

@overload
def icosahedron():
    utils3d.numpy.utils.icosahedron

@overload
def square(tri: bool = False) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Get a square mesh of area 1 centered at origin in the xy-plane.

### Returns
    vertices (np.ndarray): shape (4, 3)
    faces (np.ndarray): shape (1, 4)"""
    utils3d.numpy.utils.square

@overload
def camera_frustum(extrinsics: numpy_.ndarray, intrinsics: numpy_.ndarray, depth: float = 1.0) -> Tuple[numpy_.ndarray, numpy_.ndarray, numpy_.ndarray]:
    """Get x triangle mesh of camera frustum."""
    utils3d.numpy.utils.camera_frustum

@overload
def perspective(fov_y: Union[float, numpy_.ndarray], aspect: Union[float, numpy_.ndarray], near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Get OpenGL perspective matrix

Args:
    fov_y (float | np.ndarray): field of view in y axis
    aspect (float | np.ndarray): aspect ratio
    near (float | np.ndarray): near plane to clip
    far (float | np.ndarray): far plane to clip

Returns:
    (np.ndarray): [..., 4, 4] perspective matrix"""
    utils3d.numpy.transforms.perspective

@overload
def perspective_from_fov(fov: Union[float, numpy_.ndarray], width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray], near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Get OpenGL perspective matrix from field of view in largest dimension

Args:
    fov (float | np.ndarray): field of view in largest dimension
    width (int | np.ndarray): image width
    height (int | np.ndarray): image height
    near (float | np.ndarray): near plane to clip
    far (float | np.ndarray): far plane to clip

Returns:
    (np.ndarray): [..., 4, 4] perspective matrix"""
    utils3d.numpy.transforms.perspective_from_fov

@overload
def perspective_from_fov_xy(fov_x: Union[float, numpy_.ndarray], fov_y: Union[float, numpy_.ndarray], near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Get OpenGL perspective matrix from field of view in x and y axis

Args:
    fov_x (float | np.ndarray): field of view in x axis
    fov_y (float | np.ndarray): field of view in y axis
    near (float | np.ndarray): near plane to clip
    far (float | np.ndarray): far plane to clip

Returns:
    (np.ndarray): [..., 4, 4] perspective matrix"""
    utils3d.numpy.transforms.perspective_from_fov_xy

@overload
def intrinsics_from_focal_center(fx: Union[float, numpy_.ndarray], fy: Union[float, numpy_.ndarray], cx: Union[float, numpy_.ndarray], cy: Union[float, numpy_.ndarray], dtype: Optional[numpy_.dtype] = numpy_.float32) -> numpy_.ndarray:
    """Get OpenCV intrinsics matrix

Returns:
    (np.ndarray): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.numpy.transforms.intrinsics_from_focal_center

@overload
def intrinsics_from_fov(fov_max: Union[float, numpy_.ndarray] = None, fov_min: Union[float, numpy_.ndarray] = None, fov_x: Union[float, numpy_.ndarray] = None, fov_y: Union[float, numpy_.ndarray] = None, width: Union[int, numpy_.ndarray] = None, height: Union[int, numpy_.ndarray] = None) -> numpy_.ndarray:
    """Get normalized OpenCV intrinsics matrix from given field of view.
You can provide either fov_max, fov_min, fov_x or fov_y

Args:
    width (int | np.ndarray): image width
    height (int | np.ndarray): image height
    fov_max (float | np.ndarray): field of view in largest dimension
    fov_min (float | np.ndarray): field of view in smallest dimension
    fov_x (float | np.ndarray): field of view in x axis
    fov_y (float | np.ndarray): field of view in y axis

Returns:
    (np.ndarray): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.numpy.transforms.intrinsics_from_fov

@overload
def fov_to_focal(fov: numpy_.ndarray):
    utils3d.numpy.transforms.fov_to_focal

@overload
def focal_to_fov(focal: numpy_.ndarray):
    utils3d.numpy.transforms.focal_to_fov

@overload
def intrinsics_to_fov(intrinsics: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    utils3d.numpy.transforms.intrinsics_to_fov

@overload
def view_look_at(eye: numpy_.ndarray, look_at: numpy_.ndarray, up: numpy_.ndarray) -> numpy_.ndarray:
    """Get OpenGL view matrix looking at something

Args:
    eye (np.ndarray): [..., 3] the eye position
    look_at (np.ndarray): [..., 3] the position to look at
    up (np.ndarray): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

Returns:
    (np.ndarray): [..., 4, 4], view matrix"""
    utils3d.numpy.transforms.view_look_at

@overload
def extrinsics_look_at(eye: numpy_.ndarray, look_at: numpy_.ndarray, up: numpy_.ndarray) -> numpy_.ndarray:
    """Get OpenCV extrinsics matrix looking at something

Args:
    eye (np.ndarray): [..., 3] the eye position
    look_at (np.ndarray): [..., 3] the position to look at
    up (np.ndarray): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

Returns:
    (np.ndarray): [..., 4, 4], extrinsics matrix"""
    utils3d.numpy.transforms.extrinsics_look_at

@overload
def perspective_to_intrinsics(perspective: numpy_.ndarray) -> numpy_.ndarray:
    """OpenGL perspective matrix to OpenCV intrinsics

Args:
    perspective (np.ndarray): [..., 4, 4] OpenGL perspective matrix

Returns:
    (np.ndarray): shape [..., 3, 3] OpenCV intrinsics"""
    utils3d.numpy.transforms.perspective_to_intrinsics

@overload
def perspective_to_near_far(perspective: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Get near and far planes from OpenGL perspective matrix

Args:"""
    utils3d.numpy.transforms.perspective_to_near_far

@overload
def intrinsics_to_perspective(intrinsics: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """OpenCV intrinsics to OpenGL perspective matrix
NOTE: not work for tile-shifting intrinsics currently

Args:
    intrinsics (np.ndarray): [..., 3, 3] OpenCV intrinsics matrix
    near (float | np.ndarray): [...] near plane to clip
    far (float | np.ndarray): [...] far plane to clip
Returns:
    (np.ndarray): [..., 4, 4] OpenGL perspective matrix"""
    utils3d.numpy.transforms.intrinsics_to_perspective

@overload
def extrinsics_to_view(extrinsics: numpy_.ndarray) -> numpy_.ndarray:
    """OpenCV camera extrinsics to OpenGL view matrix

Args:
    extrinsics (np.ndarray): [..., 4, 4] OpenCV camera extrinsics matrix

Returns:
    (np.ndarray): [..., 4, 4] OpenGL view matrix"""
    utils3d.numpy.transforms.extrinsics_to_view

@overload
def view_to_extrinsics(view: numpy_.ndarray) -> numpy_.ndarray:
    """OpenGL view matrix to OpenCV camera extrinsics

Args:
    view (np.ndarray): [..., 4, 4] OpenGL view matrix

Returns:
    (np.ndarray): [..., 4, 4] OpenCV camera extrinsics matrix"""
    utils3d.numpy.transforms.view_to_extrinsics

@overload
def normalize_intrinsics(intrinsics: numpy_.ndarray, width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray], integer_pixel_centers: bool = True) -> numpy_.ndarray:
    """Normalize intrinsics from pixel cooridnates to uv coordinates

Args:
    intrinsics (np.ndarray): [..., 3, 3] camera intrinsics(s) to normalize
    width (int | np.ndarray): [...] image width(s)
    height (int | np.ndarray): [...] image height(s)
    integer_pixel_centers (bool): whether the integer pixel coordinates are at the center of the pixel. If False, the integer coordinates are at the left-top corner of the pixel.

Returns:
    (np.ndarray): [..., 3, 3] normalized camera intrinsics(s)"""
    utils3d.numpy.transforms.normalize_intrinsics

@overload
def crop_intrinsics(intrinsics: numpy_.ndarray, width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray], left: Union[int, numpy_.ndarray], top: Union[int, numpy_.ndarray], crop_width: Union[int, numpy_.ndarray], crop_height: Union[int, numpy_.ndarray]) -> numpy_.ndarray:
    """Evaluate the new intrinsics(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

Args:
    intrinsics (np.ndarray): [..., 3, 3] camera intrinsics(s) to crop
    width (int | np.ndarray): [...] image width(s)
    height (int | np.ndarray): [...] image height(s)
    left (int | np.ndarray): [...] left crop boundary
    top (int | np.ndarray): [...] top crop boundary
    crop_width (int | np.ndarray): [...] crop width
    crop_height (int | np.ndarray): [...] crop height

Returns:
    (np.ndarray): [..., 3, 3] cropped camera intrinsics(s)"""
    utils3d.numpy.transforms.crop_intrinsics

@overload
def pixel_to_uv(pixel: numpy_.ndarray, width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray]) -> numpy_.ndarray:
    """Args:
    pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
    width (int | np.ndarray): [...] image width(s)
    height (int | np.ndarray): [...] image height(s)

Returns:
    (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.numpy.transforms.pixel_to_uv

@overload
def pixel_to_ndc(pixel: numpy_.ndarray, width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray]) -> numpy_.ndarray:
    """Args:
    pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
    width (int | np.ndarray): [...] image width(s)
    height (int | np.ndarray): [...] image height(s)

Returns:
    (np.ndarray): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)"""
    utils3d.numpy.transforms.pixel_to_ndc

@overload
def uv_to_pixel(uv: numpy_.ndarray, width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray]) -> numpy_.ndarray:
    """Args:
    pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
    width (int | np.ndarray): [...] image width(s)
    height (int | np.ndarray): [...] image height(s)

Returns:
    (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.numpy.transforms.uv_to_pixel

@overload
def project_depth(depth: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Project linear depth to depth value in screen space

Args:
    depth (np.ndarray): [...] depth value
    near (float | np.ndarray): [...] near plane to clip
    far (float | np.ndarray): [...] far plane to clip

Returns:
    (np.ndarray): [..., 1] depth value in screen space, value ranging in [0, 1]"""
    utils3d.numpy.transforms.project_depth

@overload
def depth_buffer_to_linear(depth_buffer: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """OpenGL depth buffer to linear depth

Args:
    depth_buffer (np.ndarray): [...] depth value
    near (float | np.ndarray): [...] near plane to clip
    far (float | np.ndarray): [...] far plane to clip

Returns:
    (np.ndarray): [..., 1] linear depth"""
    utils3d.numpy.transforms.depth_buffer_to_linear

@overload
def unproject_cv(uv_coord: numpy_.ndarray, depth: numpy_.ndarray = None, extrinsics: numpy_.ndarray = None, intrinsics: numpy_.ndarray = None) -> numpy_.ndarray:
    """Unproject uv coordinates to 3D view space following the OpenCV convention

Args:
    uv_coord (np.ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    depth (np.ndarray): [..., N] depth value
    extrinsics (np.ndarray): [..., 4, 4] extrinsics matrix
    intrinsics (np.ndarray): [..., 3, 3] intrinsics matrix

Returns:
    points (np.ndarray): [..., N, 3] 3d points"""
    utils3d.numpy.transforms.unproject_cv

@overload
def unproject_gl(screen_coord: numpy_.ndarray, model: numpy_.ndarray = None, view: numpy_.ndarray = None, perspective: numpy_.ndarray = None) -> numpy_.ndarray:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

Args:
    screen_coord (np.ndarray): [..., N, 3] screen space coordinates, value ranging in [0, 1].
        The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
    model (np.ndarray): [..., 4, 4] model matrix
    view (np.ndarray): [..., 4, 4] view matrix
    perspective (np.ndarray): [..., 4, 4] perspective matrix

Returns:
    points (np.ndarray): [..., N, 3] 3d points"""
    utils3d.numpy.transforms.unproject_gl

@overload
def project_cv(points: numpy_.ndarray, extrinsics: numpy_.ndarray = None, intrinsics: numpy_.ndarray = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Project 3D points to 2D following the OpenCV convention

Args:
    points (np.ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last
        dimension is 4, the points are assumed to be in homogeneous coordinates
    extrinsics (np.ndarray): [..., 4, 4] extrinsics matrix
    intrinsics (np.ndarray): [..., 3, 3] intrinsics matrix

Returns:
    uv_coord (np.ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    linear_depth (np.ndarray): [..., N] linear depth"""
    utils3d.numpy.transforms.project_cv

@overload
def project_gl(points: numpy_.ndarray, model: numpy_.ndarray = None, view: numpy_.ndarray = None, perspective: numpy_.ndarray = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrice)

Args:
    points (np.ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
        dimension is 4, the points are assumed to be in homogeneous coordinates
    model (np.ndarray): [..., 4, 4] model matrix
    view (np.ndarray): [..., 4, 4] view matrix
    perspective (np.ndarray): [..., 4, 4] perspective matrix

Returns:
    scr_coord (np.ndarray): [..., N, 3] screen space coordinates, value ranging in [0, 1].
        The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
    linear_depth (np.ndarray): [..., N] linear depth"""
    utils3d.numpy.transforms.project_gl

@overload
def quaternion_to_matrix(quaternion: numpy_.ndarray, eps: float = 1e-12) -> numpy_.ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices

Args:
    quaternion (np.ndarray): shape (..., 4), the quaternions to convert

Returns:
    np.ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions"""
    utils3d.numpy.transforms.quaternion_to_matrix

@overload
def axis_angle_to_matrix(axis_angle: numpy_.ndarray, eps: float = 1e-12) -> numpy_.ndarray:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

Args:
    axis_angle (np.ndarray): shape (..., 3), axis-angle vcetors

Returns:
    np.ndarray: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters"""
    utils3d.numpy.transforms.axis_angle_to_matrix

@overload
def matrix_to_quaternion(rot_mat: numpy_.ndarray, eps: float = 1e-12) -> numpy_.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

Args:
    rot_mat (np.ndarray): shape (..., 3, 3), the rotation matrices to convert

Returns:
    np.ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices"""
    utils3d.numpy.transforms.matrix_to_quaternion

@overload
def extrinsics_to_essential(extrinsics: numpy_.ndarray):
    """extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

Args:
    extrinsics (np.ndaray): [..., 4, 4] extrinsics matrix

Returns:
    (np.ndaray): [..., 3, 3] essential matrix"""
    utils3d.numpy.transforms.extrinsics_to_essential

@overload
def euler_axis_angle_rotation(axis: str, angle: numpy_.ndarray) -> numpy_.ndarray:
    """Return the rotation matrices for one of the rotations about an axis
of which Euler angles describe, for each value of the angle given.

Args:
    axis: Axis label "X" or "Y or "Z".
    angle: any shape tensor of Euler angles in radians

Returns:
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.numpy.transforms.euler_axis_angle_rotation

@overload
def euler_angles_to_matrix(euler_angles: numpy_.ndarray, convention: str = 'XYZ') -> numpy_.ndarray:
    """Convert rotations given as Euler angles in radians to rotation matrices.

Args:
    euler_angles: Euler angles in radians as ndarray of shape (..., 3), XYZ
    convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

Returns:
    Rotation matrices as ndarray of shape (..., 3, 3)."""
    utils3d.numpy.transforms.euler_angles_to_matrix

@overload
def skew_symmetric(v: numpy_.ndarray):
    """Skew symmetric matrix from a 3D vector"""
    utils3d.numpy.transforms.skew_symmetric

@overload
def rotation_matrix_from_vectors(v1: numpy_.ndarray, v2: numpy_.ndarray):
    """Rotation matrix that rotates v1 to v2"""
    utils3d.numpy.transforms.rotation_matrix_from_vectors

@overload
def ray_intersection(p1: numpy_.ndarray, d1: numpy_.ndarray, p2: numpy_.ndarray, d2: numpy_.ndarray):
    """Compute the intersection/closest point of two D-dimensional rays
If the rays are intersecting, the closest point is the intersection point.

Args:
    p1 (np.ndarray): (..., D) origin of ray 1
    d1 (np.ndarray): (..., D) direction of ray 1
    p2 (np.ndarray): (..., D) origin of ray 2
    d2 (np.ndarray): (..., D) direction of ray 2

Returns:
    (np.ndarray): (..., N) intersection point"""
    utils3d.numpy.transforms.ray_intersection

@overload
def se3_matrix(R: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Convert rotation matrix and translation vector to 4x4 transformation matrix.

Args:
    R (np.ndarray): [..., 3, 3] rotation matrix
    t (np.ndarray): [..., 3] translation vector

Returns:
    np.ndarray: [..., 4, 4] transformation matrix"""
    utils3d.numpy.transforms.se3_matrix

@overload
def slerp_quaternion(q1: numpy_.ndarray, q2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Spherical linear interpolation between two unit quaternions.

Args:
    q1 (np.ndarray): [..., d] unit vector 1
    q2 (np.ndarray): [..., d] unit vector 2
    t (np.ndarray): [...] interpolation parameter in [0, 1]

Returns:
    np.ndarray: [..., 3] interpolated unit vector"""
    utils3d.numpy.transforms.slerp_quaternion

@overload
def slerp_vector(v1: numpy_.ndarray, v2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Spherical linear interpolation between two unit vectors. The vectors are assumed to be normalized.

Args:
    v1 (np.ndarray): [..., d] unit vector 1
    v2 (np.ndarray): [..., d] unit vector 2
    t (np.ndarray): [...] interpolation parameter in [0, 1]

Returns:
    np.ndarray: [..., d] interpolated unit vector"""
    utils3d.numpy.transforms.slerp_vector

@overload
def lerp(x1: numpy_.ndarray, x2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Linear interpolation between two vectors.

Args:
    x1 (np.ndarray): [..., d] vector 1
    x2 (np.ndarray): [..., d] vector 2
    t (np.ndarray): [...] interpolation parameter. [0, 1] for interpolation between x1 and x2, otherwise for extrapolation.

Returns:
    np.ndarray: [..., d] interpolated vector"""
    utils3d.numpy.transforms.lerp

@overload
def lerp_se3_matrix(T1: numpy_.ndarray, T2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Linear interpolation between two SE(3) matrices.

Args:
    T1 (np.ndarray): [..., 4, 4] SE(3) matrix 1
    T2 (np.ndarray): [..., 4, 4] SE(3) matrix 2
    t (np.ndarray): [...] interpolation parameter in [0, 1]

Returns:
    np.ndarray: [..., 4, 4] interpolated SE(3) matrix"""
    utils3d.numpy.transforms.lerp_se3_matrix

@overload
def piecewise_lerp(x: numpy_.ndarray, t: numpy_.ndarray, s: numpy_.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> numpy_.ndarray:
    """Linear spline interpolation.

### Parameters:
- `x`: np.ndarray, shape (n, d): the values of data points.
- `t`: np.ndarray, shape (n,): the times of the data points.
- `s`: np.ndarray, shape (m,): the times to be interpolated.
- `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

### Returns:
- `y`: np.ndarray, shape (..., m, d): the interpolated values."""
    utils3d.numpy.transforms.piecewise_lerp

@overload
def piecewise_lerp_se3_matrix(T: numpy_.ndarray, t: numpy_.ndarray, s: numpy_.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> numpy_.ndarray:
    """Linear spline interpolation for SE(3) matrices.

### Parameters:
- `T`: np.ndarray, shape (n, 4, 4): the SE(3) matrices.
- `t`: np.ndarray, shape (n,): the times of the data points.
- `s`: np.ndarray, shape (m,): the times to be interpolated.
- `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

### Returns:
- `T_interp`: np.ndarray, shape (..., m, 4, 4): the interpolated SE(3) matrices."""
    utils3d.numpy.transforms.piecewise_lerp_se3_matrix

@overload
def apply_transform(T: numpy_.ndarray, x: numpy_.ndarray) -> numpy_.ndarray:
    """Apply SE(3) transformation to a point or a set of points.

### Parameters:
- `T`: np.ndarray, shape (..., 4, 4): the SE(3) matrix.
- `x`: np.ndarray, shape (..., 3): the point or a set of points to be transformed.

### Returns:
- `x_transformed`: np.ndarray, shape (..., 3): the transformed point or a set of points."""
    utils3d.numpy.transforms.apply_transform

@overload
def linear_spline_interpolate(x: numpy_.ndarray, t: numpy_.ndarray, s: numpy_.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> numpy_.ndarray:
    """Linear spline interpolation.

### Parameters:
- `x`: np.ndarray, shape (n, d): the values of data points.
- `t`: np.ndarray, shape (n,): the times of the data points.
- `s`: np.ndarray, shape (m,): the times to be interpolated.
- `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

### Returns:
- `y`: np.ndarray, shape (..., m, d): the interpolated values."""
    utils3d.numpy.spline.linear_spline_interpolate

@overload
def RastContext(*args, **kwargs):
    utils3d.numpy.rasterization.RastContext

@overload
def rasterize_triangle_faces(ctx: utils3d.numpy.rasterization.RastContext, vertices: numpy_.ndarray, faces: numpy_.ndarray, attr: numpy_.ndarray, width: int, height: int, transform: numpy_.ndarray = None, cull_backface: bool = True, return_depth: bool = False, image: numpy_.ndarray = None, depth: numpy_.ndarray = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Rasterize vertex attribute.

Args:
    vertices (np.ndarray): [N, 3]
    faces (np.ndarray): [T, 3]
    attr (np.ndarray): [N, C]
    width (int): width of rendered image
    height (int): height of rendered image
    transform (np.ndarray): [4, 4] model-view-projection transformation matrix. 
    cull_backface (bool): whether to cull backface
    image: (np.ndarray): [H, W, C] background image
    depth: (np.ndarray): [H, W] background depth

Returns:
    image (np.ndarray): [H, W, C] rendered image
    depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None."""
    utils3d.numpy.rasterization.rasterize_triangle_faces

@overload
def rasterize_edges(ctx: utils3d.numpy.rasterization.RastContext, vertices: numpy_.ndarray, edges: numpy_.ndarray, attr: numpy_.ndarray, width: int, height: int, transform: numpy_.ndarray = None, line_width: float = 1.0, return_depth: bool = False, image: numpy_.ndarray = None, depth: numpy_.ndarray = None) -> Tuple[numpy_.ndarray, ...]:
    """Rasterize vertex attribute.

Args:
    vertices (np.ndarray): [N, 3]
    faces (np.ndarray): [T, 3]
    attr (np.ndarray): [N, C]
    width (int): width of rendered image
    height (int): height of rendered image
    transform (np.ndarray): [4, 4] model-view-projection matrix
    line_width (float): width of line. Defaults to 1.0. NOTE: Values other than 1.0 may not work across all platforms.
    cull_backface (bool): whether to cull backface

Returns:
    image (np.ndarray): [H, W, C] rendered image
    depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None."""
    utils3d.numpy.rasterization.rasterize_edges

@overload
def texture(ctx: utils3d.numpy.rasterization.RastContext, uv: numpy_.ndarray, texture: numpy_.ndarray, interpolation: str = 'linear', wrap: str = 'clamp') -> numpy_.ndarray:
    """Given an UV image, texturing from the texture map"""
    utils3d.numpy.rasterization.texture

@overload
def warp_image_by_depth(ctx: utils3d.numpy.rasterization.RastContext, src_depth: numpy_.ndarray, src_image: numpy_.ndarray = None, width: int = None, height: int = None, *, extrinsics_src: numpy_.ndarray = None, extrinsics_tgt: numpy_.ndarray = None, intrinsics_src: numpy_.ndarray = None, intrinsics_tgt: numpy_.ndarray = None, near: float = 0.1, far: float = 100.0, cull_backface: bool = True, ssaa: int = 1, return_depth: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Warp image by depth map.

Args:
    ctx (RastContext): rasterizer context
    src_depth (np.ndarray): [H, W]
    src_image (np.ndarray, optional): [H, W, C]. The image to warp. Defaults to None (use uv coordinates).
    width (int, optional): width of the output image. None to use depth map width. Defaults to None.
    height (int, optional): height of the output image. None to use depth map height. Defaults to None.
    extrinsics_src (np.ndarray, optional): extrinsics matrix of the source camera. Defaults to None (identity).
    extrinsics_tgt (np.ndarray, optional): extrinsics matrix of the target camera. Defaults to None (identity).
    intrinsics_src (np.ndarray, optional): intrinsics matrix of the source camera. Defaults to None (use the same as intrinsics_tgt).
    intrinsics_tgt (np.ndarray, optional): intrinsics matrix of the target camera. Defaults to None (use the same as intrinsics_src).
    cull_backface (bool, optional): whether to cull backface. Defaults to True.
    ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.

Returns:
    tgt_image (np.ndarray): [H, W, C] warped image (or uv coordinates if image is None).
    tgt_depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None."""
    utils3d.numpy.rasterization.warp_image_by_depth

@overload
def test_rasterization(ctx: utils3d.numpy.rasterization.RastContext):
    """Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file."""
    utils3d.numpy.rasterization.test_rasterization

@overload
def triangulate(faces: torch_.Tensor, vertices: torch_.Tensor = None, backslash: bool = None) -> torch_.Tensor:
    """Triangulate a polygonal mesh.

Args:
    faces (torch.Tensor): [..., L, P] polygonal faces
    vertices (torch.Tensor, optional): [..., N, 3] 3-dimensional vertices.
        If given, the triangulation is performed according to the distance
        between vertices. Defaults to None.
    backslash (torch.Tensor, optional): [..., L] boolean array indicating
        how to triangulate the quad faces. Defaults to None.


Returns:
    (torch.Tensor): [L * (P - 2), 3] triangular faces"""
    utils3d.torch.mesh.triangulate

@overload
def compute_face_normal(vertices: torch_.Tensor, faces: torch_.Tensor) -> torch_.Tensor:
    """Compute face normals of a triangular mesh

Args:
    vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
    faces (torch.Tensor): [..., T, 3] triangular face indices

Returns:
    normals (torch.Tensor): [..., T, 3] face normals"""
    utils3d.torch.mesh.compute_face_normal

@overload
def compute_face_angles(vertices: torch_.Tensor, faces: torch_.Tensor) -> torch_.Tensor:
    """Compute face angles of a triangular mesh

Args:
    vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
    faces (torch.Tensor): [T, 3] triangular face indices

Returns:
    angles (torch.Tensor): [..., T, 3] face angles"""
    utils3d.torch.mesh.compute_face_angles

@overload
def compute_vertex_normal(vertices: torch_.Tensor, faces: torch_.Tensor, face_normal: torch_.Tensor = None) -> torch_.Tensor:
    """Compute vertex normals of a triangular mesh by averaging neightboring face normals

Args:
    vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
    faces (torch.Tensor): [T, 3] triangular face indices
    face_normal (torch.Tensor, optional): [..., T, 3] face normals.
        None to compute face normals from vertices and faces. Defaults to None.

Returns:
    normals (torch.Tensor): [..., N, 3] vertex normals"""
    utils3d.torch.mesh.compute_vertex_normal

@overload
def compute_vertex_normal_weighted(vertices: torch_.Tensor, faces: torch_.Tensor, face_normal: torch_.Tensor = None) -> torch_.Tensor:
    """Compute vertex normals of a triangular mesh by weighted sum of neightboring face normals
according to the angles

Args:
    vertices (torch.Tensor): [..., N, 3] 3-dimensional vertices
    faces (torch.Tensor): [T, 3] triangular face indices
    face_normal (torch.Tensor, optional): [..., T, 3] face normals.
        None to compute face normals from vertices and faces. Defaults to None.

Returns:
    normals (torch.Tensor): [..., N, 3] vertex normals"""
    utils3d.torch.mesh.compute_vertex_normal_weighted

@overload
def remove_unreferenced_vertices(faces: torch_.Tensor, *vertice_attrs, return_indices: bool = False) -> Tuple[torch_.Tensor, ...]:
    """Remove unreferenced vertices of a mesh. 
Unreferenced vertices are removed, and the face indices are updated accordingly.

Args:
    faces (torch.Tensor): [T, P] face indices
    *vertice_attrs: vertex attributes

Returns:
    faces (torch.Tensor): [T, P] face indices
    *vertice_attrs: vertex attributes
    indices (torch.Tensor, optional): [N] indices of vertices that are kept. Defaults to None."""
    utils3d.torch.mesh.remove_unreferenced_vertices

@overload
def remove_corrupted_faces(faces: torch_.Tensor) -> torch_.Tensor:
    """Remove corrupted faces (faces with duplicated vertices)

Args:
    faces (torch.Tensor): [T, 3] triangular face indices

Returns:
    torch.Tensor: [T_, 3] triangular face indices"""
    utils3d.torch.mesh.remove_corrupted_faces

@overload
def merge_duplicate_vertices(vertices: torch_.Tensor, faces: torch_.Tensor, tol: float = 1e-06) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Merge duplicate vertices of a triangular mesh. 
Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

Args:
    vertices (torch.Tensor): [N, 3] 3-dimensional vertices
    faces (torch.Tensor): [T, 3] triangular face indices
    tol (float, optional): tolerance for merging. Defaults to 1e-6.

Returns:
    vertices (torch.Tensor): [N_, 3] 3-dimensional vertices
    faces (torch.Tensor): [T, 3] triangular face indices"""
    utils3d.torch.mesh.merge_duplicate_vertices

@overload
def subdivide_mesh_simple(vertices: torch_.Tensor, faces: torch_.Tensor, n: int = 1) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.

Args:
    vertices (torch.Tensor): [N, 3] 3-dimensional vertices
    faces (torch.Tensor): [T, 3] triangular face indices
    n (int, optional): number of subdivisions. Defaults to 1.

Returns:
    vertices (torch.Tensor): [N_, 3] subdivided 3-dimensional vertices
    faces (torch.Tensor): [4 * T, 3] subdivided triangular face indices"""
    utils3d.torch.mesh.subdivide_mesh_simple

@overload
def compute_face_tbn(pos: torch_.Tensor, faces_pos: torch_.Tensor, uv: torch_.Tensor, faces_uv: torch_.Tensor, eps: float = 1e-07) -> torch_.Tensor:
    """compute TBN matrix for each face

Args:
    pos (torch.Tensor): shape (..., N_pos, 3), positions
    faces_pos (torch.Tensor): shape(T, 3) 
    uv (torch.Tensor): shape (..., N_uv, 3) uv coordinates, 
    faces_uv (torch.Tensor): shape(T, 3) 
    
Returns:
    torch.Tensor: (..., T, 3, 3) TBN matrix for each face. Note TBN vectors are normalized but not necessarily orthognal"""
    utils3d.torch.mesh.compute_face_tbn

@overload
def compute_vertex_tbn(faces_topo: torch_.Tensor, pos: torch_.Tensor, faces_pos: torch_.Tensor, uv: torch_.Tensor, faces_uv: torch_.Tensor) -> torch_.Tensor:
    """compute TBN matrix for each face

Args:
    faces_topo (torch.Tensor): (T, 3), face indice of topology
    pos (torch.Tensor): shape (..., N_pos, 3), positions
    faces_pos (torch.Tensor): shape(T, 3) 
    uv (torch.Tensor): shape (..., N_uv, 3) uv coordinates, 
    faces_uv (torch.Tensor): shape(T, 3) 
    
Returns:
    torch.Tensor: (..., V, 3, 3) TBN matrix for each face. Note TBN vectors are normalized but not necessarily orthognal"""
    utils3d.torch.mesh.compute_vertex_tbn

@overload
def laplacian(vertices: torch_.Tensor, faces: torch_.Tensor, weight: str = 'uniform') -> torch_.Tensor:
    """Laplacian smooth with cotangent weights

Args:
    vertices (torch.Tensor): shape (..., N, 3)
    faces (torch.Tensor): shape (T, 3)
    weight (str): 'uniform' or 'cotangent'"""
    utils3d.torch.mesh.laplacian

@overload
def laplacian_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, weight: str = 'uniform', times: int = 5) -> torch_.Tensor:
    """Laplacian smooth with cotangent weights

Args:
    vertices (torch.Tensor): shape (..., N, 3)
    faces (torch.Tensor): shape (T, 3)
    weight (str): 'uniform' or 'cotangent'"""
    utils3d.torch.mesh.laplacian_smooth_mesh

@overload
def taubin_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, lambda_: float = 0.5, mu_: float = -0.51) -> torch_.Tensor:
    """Taubin smooth mesh

Args:
    vertices (torch.Tensor): _description_
    faces (torch.Tensor): _description_
    lambda_ (float, optional): _description_. Defaults to 0.5.
    mu_ (float, optional): _description_. Defaults to -0.51.

Returns:
    torch.Tensor: _description_"""
    utils3d.torch.mesh.taubin_smooth_mesh

@overload
def laplacian_hc_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, times: int = 5, alpha: float = 0.5, beta: float = 0.5, weight: str = 'uniform'):
    """HC algorithm from Improved Laplacian Smoothing of Noisy Surface Meshes by J.Vollmer et al.
    """
    utils3d.torch.mesh.laplacian_hc_smooth_mesh

@overload
def get_rays(extrinsics: torch_.Tensor, intrinsics: torch_.Tensor, uv: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Args:
    extrinsics: (..., 4, 4) extrinsics matrices.
    intrinsics: (..., 3, 3) intrinsics matrices.
    uv: (..., n_rays, 2) uv coordinates of the rays. 

Returns:
    rays_o: (..., 1,      3) ray origins
    rays_d: (..., n_rays, 3) ray directions. 
        NOTE: ray directions are NOT normalized. They actuallys makes rays_o + rays_d * z = world coordinates, where z is the depth."""
    utils3d.torch.nerf.get_rays

@overload
def get_image_rays(extrinsics: torch_.Tensor, intrinsics: torch_.Tensor, width: int, height: int) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Args:
    extrinsics: (..., 4, 4) extrinsics matrices.
    intrinsics: (..., 3, 3) intrinsics matrices.
    width: width of the image.
    height: height of the image.

Returns:
    rays_o: (..., 1,      1,     3) ray origins
    rays_d: (..., height, width, 3) ray directions. 
        NOTE: ray directions are NOT normalized. They actuallys makes rays_o + rays_d * z = world coordinates, where z is the depth."""
    utils3d.torch.nerf.get_image_rays

@overload
def get_mipnerf_cones(rays_o: torch_.Tensor, rays_d: torch_.Tensor, z_vals: torch_.Tensor, pixel_width: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Args:
    rays_o: (..., n_rays, 3) ray origins
    rays_d: (..., n_rays, 3) ray directions.
    z_vals: (..., n_rays, n_samples) z values.
    pixel_width: (...) pixel width. = 1 / (normalized focal length * width)

Returns:
    mu: (..., n_rays, n_samples, 3) cone mu.
    sigma: (..., n_rays, n_samples, 3, 3) cone sigma."""
    utils3d.torch.nerf.get_mipnerf_cones

@overload
def volume_rendering(color: torch_.Tensor, sigma: torch_.Tensor, z_vals: torch_.Tensor, ray_length: torch_.Tensor, rgb: bool = True, depth: bool = True) -> Tuple[torch_.Tensor, torch_.Tensor, torch_.Tensor]:
    """Given color, sigma and z_vals (linear depth of the sampling points), render the volume.

NOTE: By default, color and sigma should have one less sample than z_vals, in correspondence with the average value in intervals.
If queried color are aligned with z_vals, we use trapezoidal rule to calculate the average values in intervals.

Args:
    color: (..., n_samples or n_samples - 1, 3) color values.
    sigma: (..., n_samples or n_samples - 1) density values.
    z_vals: (..., n_samples) z values.
    ray_length: (...) length of the ray

Returns:
    rgb: (..., 3) rendered color values.
    depth: (...) rendered depth values.
    weights (..., n_samples) weights."""
    utils3d.torch.nerf.volume_rendering

@overload
def bin_sample(size: Union[torch_.Size, Tuple[int, ...]], n_samples: int, min_value: numbers.Number, max_value: numbers.Number, spacing: Literal['linear', 'inverse_linear'], dtype: torch_.dtype = None, device: torch_.device = None) -> torch_.Tensor:
    """Uniformly (or uniformly in inverse space) sample z values in `n_samples` bins in range [min_value, max_value].
Args:
    size: size of the rays
    n_samples: number of samples to be sampled, also the number of bins
    min_value: minimum value of the range
    max_value: maximum value of the range
    space: 'linear' or 'inverse_linear'. If 'inverse_linear', the sampling is uniform in inverse space.

Returns:
    z_rand: (*size, n_samples) sampled z values, sorted in ascending order."""
    utils3d.torch.nerf.bin_sample

@overload
def importance_sample(z_vals: torch_.Tensor, weights: torch_.Tensor, n_samples: int) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Importance sample z values.

NOTE: By default, weights should have one less sample than z_vals, in correspondence with the intervals.
If weights has the same number of samples as z_vals, we use trapezoidal rule to calculate the average weights in intervals.

Args:
    z_vals: (..., n_rays, n_input_samples) z values, sorted in ascending order.
    weights: (..., n_rays, n_input_samples or n_input_samples - 1) weights.
    n_samples: number of output samples for importance sampling.

Returns:
    z_importance: (..., n_rays, n_samples) importance sampled z values, unsorted."""
    utils3d.torch.nerf.importance_sample

@overload
def nerf_render_rays(nerf: Union[Callable[[torch_.Tensor, torch_.Tensor], Tuple[torch_.Tensor, torch_.Tensor]], Tuple[Callable[[torch_.Tensor], Tuple[torch_.Tensor, torch_.Tensor]], Callable[[torch_.Tensor], Tuple[torch_.Tensor, torch_.Tensor]]]], rays_o: torch_.Tensor, rays_d: torch_.Tensor, *, return_dict: bool = False, n_coarse: int = 64, n_fine: int = 64, near: float = 0.1, far: float = 100.0, z_spacing: Literal['linear', 'inverse_linear'] = 'linear'):
    """NeRF rendering of rays. Note that it supports arbitrary batch dimensions (denoted as `...`)

Args:
    nerf: nerf model, which takes (points, directions) as input and returns (color, density) as output.
        If nerf is a tuple, it should be (nerf_coarse, nerf_fine), where nerf_coarse and nerf_fine are two nerf models for coarse and fine stages respectively.
        
        nerf args:
            points: (..., n_rays, n_samples, 3)
            directions: (..., n_rays, n_samples, 3)
        nerf returns:
            color: (..., n_rays, n_samples, 3) color values.
            density: (..., n_rays, n_samples) density values.
            
    rays_o: (..., n_rays, 3) ray origins
    rays_d: (..., n_rays, 3) ray directions.
    pixel_width: (..., n_rays) pixel width. How to compute? pixel_width = 1 / (normalized focal length * width)

Returns 
    if return_dict is False, return rendered rgb and depth for short cut. (If there are separate coarse and fine results, return fine results)
        rgb: (..., n_rays, 3) rendered color values. 
        depth: (..., n_rays) rendered depth values.
    else, return a dict. If `n_fine == 0` or `nerf` is a single model, the dict only contains coarse results:
    ```
    {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
    ```
    If there are two models for coarse and fine stages, the dict contains both coarse and fine results:
    ```
    {
        "coarse": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..},
        "fine": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
    }
    ```"""
    utils3d.torch.nerf.nerf_render_rays

@overload
def mipnerf_render_rays(mipnerf: Callable[[torch_.Tensor, torch_.Tensor, torch_.Tensor], Tuple[torch_.Tensor, torch_.Tensor]], rays_o: torch_.Tensor, rays_d: torch_.Tensor, pixel_width: torch_.Tensor, *, return_dict: bool = False, n_coarse: int = 64, n_fine: int = 64, uniform_ratio: float = 0.4, near: float = 0.1, far: float = 100.0, z_spacing: Literal['linear', 'inverse_linear'] = 'linear') -> Union[Tuple[torch_.Tensor, torch_.Tensor], Dict[str, torch_.Tensor]]:
    """MipNeRF rendering.

Args:
    mipnerf: mipnerf model, which takes (points_mu, points_sigma) as input and returns (color, density) as output.

        mipnerf args:
            points_mu: (..., n_rays, n_samples, 3) cone mu.
            points_sigma: (..., n_rays, n_samples, 3, 3) cone sigma.
            directions: (..., n_rays, n_samples, 3)
        mipnerf returns:
            color: (..., n_rays, n_samples, 3) color values.
            density: (..., n_rays, n_samples) density values.

    rays_o: (..., n_rays, 3) ray origins
    rays_d: (..., n_rays, 3) ray directions.
    pixel_width: (..., n_rays) pixel width. How to compute? pixel_width = 1 / (normalized focal length * width)

Returns 
    if return_dict is False, return rendered results only: (If `n_fine == 0`, return coarse results, otherwise return fine results)
        rgb: (..., n_rays, 3) rendered color values. 
        depth: (..., n_rays) rendered depth values.
    else, return a dict. If `n_fine == 0`, the dict only contains coarse results:
    ```
    {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
    ```
    If n_fine > 0, the dict contains both coarse and fine results :
    ```
    {
        "coarse": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..},
        "fine": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
    }
    ```"""
    utils3d.torch.nerf.mipnerf_render_rays

@overload
def nerf_render_view(nerf: torch_.Tensor, extrinsics: torch_.Tensor, intrinsics: torch_.Tensor, width: int, height: int, *, patchify: bool = False, patch_size: Tuple[int, int] = (64, 64), **options: Dict[str, Any]) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """NeRF rendering of views. Note that it supports arbitrary batch dimensions (denoted as `...`)

Args:
    extrinsics: (..., 4, 4) extrinsics matrice of the rendered views
    intrinsics (optional): (..., 3, 3) intrinsics matrice of the rendered views.
    width (optional): image width of the rendered views.
    height (optional): image height of the rendered views.
    patchify (optional): If the image is too large, render it patch by patch
    **options: rendering options.

Returns:
    rgb: (..., channels, height, width) rendered color values.
    depth: (..., height, width) rendered depth values."""
    utils3d.torch.nerf.nerf_render_view

@overload
def mipnerf_render_view(mipnerf: torch_.Tensor, extrinsics: torch_.Tensor, intrinsics: torch_.Tensor, width: int, height: int, *, patchify: bool = False, patch_size: Tuple[int, int] = (64, 64), **options: Dict[str, Any]) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """MipNeRF rendering of views. Note that it supports arbitrary batch dimensions (denoted as `...`)

Args:
    extrinsics: (..., 4, 4) extrinsics matrice of the rendered views
    intrinsics (optional): (..., 3, 3) intrinsics matrice of the rendered views.
    width (optional): image width of the rendered views.
    height (optional): image height of the rendered views.
    patchify (optional): If the image is too large, render it patch by patch
    **options: rendering options.

Returns:
    rgb: (..., 3, height, width) rendered color values.
    depth: (..., height, width) rendered depth values."""
    utils3d.torch.nerf.mipnerf_render_view

@overload
def InstantNGP(view_dependent: bool = True, base_resolution: int = 16, finest_resolution: int = 2048, n_levels: int = 16, num_layers_density: int = 2, hidden_dim_density: int = 64, num_layers_color: int = 3, hidden_dim_color: int = 64, log2_hashmap_size: int = 19, bound: float = 1.0, color_channels: int = 3):
    """An implementation of InstantNGP, Müller et. al., https://nvlabs.github.io/instant-ngp/.
Requires `tinycudann` package.
Install it by:
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```"""
    utils3d.torch.nerf.InstantNGP

@overload
def sliding_window_1d(x: torch_.Tensor, window_size: int, stride: int = 1, dim: int = -1) -> torch_.Tensor:
    """Sliding window view of the input tensor. The dimension of the sliding window is appended to the end of the input tensor's shape.
NOTE: Since Pytorch has `unfold` function, 1D sliding window view is just a wrapper of it."""
    utils3d.torch.utils.sliding_window_1d

@overload
def sliding_window_2d(x: torch_.Tensor, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dim: Union[int, Tuple[int, int]] = (-2, -1)) -> torch_.Tensor:
    utils3d.torch.utils.sliding_window_2d

@overload
def sliding_window_nd(x: torch_.Tensor, window_size: Tuple[int, ...], stride: Tuple[int, ...], dim: Tuple[int, ...]) -> torch_.Tensor:
    utils3d.torch.utils.sliding_window_nd

@overload
def image_uv(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch_.device = None, dtype: torch_.dtype = None) -> torch_.Tensor:
    """Get image space UV grid, ranging in [0, 1]. 

>>> image_uv(10, 10):
[[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
 [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
  ...             ...                  ...
 [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

Args:
    width (int): image width
    height (int): image height

Returns:
    np.ndarray: shape (height, width, 2)"""
    utils3d.torch.utils.image_uv

@overload
def image_pixel_center(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, dtype: torch_.dtype = None, device: torch_.device = None) -> torch_.Tensor:
    """Get image pixel center coordinates, ranging in [0, width] and [0, height].
`image[i, j]` has pixel center coordinates `(j + 0.5, i + 0.5)`.

>>> image_pixel_center(10, 10):
[[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
 [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
  ...             ...                  ...
[[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]

Args:
    width (int): image width
    height (int): image height

Returns:
    np.ndarray: shape (height, width, 2)"""
    utils3d.torch.utils.image_pixel_center

@overload
def image_mesh(height: int, width: int, mask: torch_.Tensor = None, device: torch_.device = None, dtype: torch_.dtype = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Get a quad mesh regarding image pixel uv coordinates as vertices and image grid as faces.

Args:
    width (int): image width
    height (int): image height
    mask (np.ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

Returns:
    uv (np.ndarray): uv corresponding to pixels as described in image_uv()
    faces (np.ndarray): quad faces connecting neighboring pixels
    indices (np.ndarray, optional): indices of vertices in the original mesh"""
    utils3d.torch.utils.image_mesh

@overload
def chessboard(width: int, height: int, grid_size: int, color_a: torch_.Tensor, color_b: torch_.Tensor) -> torch_.Tensor:
    """get a chessboard image

Args:
    width (int): image width
    height (int): image height
    grid_size (int): size of chessboard grid
    color_a (torch.Tensor): shape (chanenls,), color of the grid at the top-left corner
    color_b (torch.Tensor): shape (chanenls,), color in complementary grids

Returns:
    image (torch.Tensor): shape (height, width, channels), chessboard image"""
    utils3d.torch.utils.chessboard

@overload
def depth_edge(depth: torch_.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch_.Tensor = None) -> torch_.BoolTensor:
    """Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.

Args:
    depth (torch.Tensor): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

Returns:
    edge (torch.Tensor): shape (..., height, width) of dtype torch.bool"""
    utils3d.torch.utils.depth_edge

@overload
def depth_aliasing(depth: torch_.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch_.Tensor = None) -> torch_.BoolTensor:
    """Compute the map that indicates the aliasing of a depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
Args:
    depth (torch.Tensor): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

Returns:
    edge (torch.Tensor): shape (..., height, width) of dtype torch.bool"""
    utils3d.torch.utils.depth_aliasing

@overload
def image_mesh_from_depth(depth: torch_.Tensor, extrinsics: torch_.Tensor = None, intrinsics: torch_.Tensor = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    utils3d.torch.utils.image_mesh_from_depth

@overload
def point_to_normal(point: torch_.Tensor, mask: torch_.Tensor = None) -> torch_.Tensor:
    """Calculate normal map from point map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

Args:
    point (torch.Tensor): shape (..., height, width, 3), point map
Returns:
    normal (torch.Tensor): shape (..., height, width, 3), normal map. """
    utils3d.torch.utils.point_to_normal

@overload
def depth_to_normal(depth: torch_.Tensor, intrinsics: torch_.Tensor, mask: torch_.Tensor = None) -> torch_.Tensor:
    """Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

Args:
    depth (torch.Tensor): shape (..., height, width), linear depth map
    intrinsics (torch.Tensor): shape (..., 3, 3), intrinsics matrix
Returns:
    normal (torch.Tensor): shape (..., 3, height, width), normal map. """
    utils3d.torch.utils.depth_to_normal

@overload
def masked_min(input: torch_.Tensor, mask: torch_.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch_.Tensor, Tuple[torch_.Tensor, torch_.Tensor]]:
    """Similar to torch.min, but with mask
    """
    utils3d.torch.utils.masked_min

@overload
def masked_max(input: torch_.Tensor, mask: torch_.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch_.Tensor, Tuple[torch_.Tensor, torch_.Tensor]]:
    """Similar to torch.max, but with mask
    """
    utils3d.torch.utils.masked_max

@overload
def bounding_rect(mask: torch_.BoolTensor):
    """get bounding rectangle of a mask

Args:
    mask (torch.Tensor): shape (..., height, width), mask

Returns:
    rect (torch.Tensor): shape (..., 4), bounding rectangle (left, top, right, bottom)"""
    utils3d.torch.utils.bounding_rect

@overload
def perspective(fov_y: Union[float, torch_.Tensor], aspect: Union[float, torch_.Tensor], near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenGL perspective matrix

Args:
    fov_y (float | torch.Tensor): field of view in y axis
    aspect (float | torch.Tensor): aspect ratio
    near (float | torch.Tensor): near plane to clip
    far (float | torch.Tensor): far plane to clip

Returns:
    (torch.Tensor): [..., 4, 4] perspective matrix"""
    utils3d.torch.transforms.perspective

@overload
def perspective_from_fov(fov: Union[float, torch_.Tensor], width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor], near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenGL perspective matrix from field of view in largest dimension

Args:
    fov (float | torch.Tensor): field of view in largest dimension
    width (int | torch.Tensor): image width
    height (int | torch.Tensor): image height
    near (float | torch.Tensor): near plane to clip
    far (float | torch.Tensor): far plane to clip

Returns:
    (torch.Tensor): [..., 4, 4] perspective matrix"""
    utils3d.torch.transforms.perspective_from_fov

@overload
def perspective_from_fov_xy(fov_x: Union[float, torch_.Tensor], fov_y: Union[float, torch_.Tensor], near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenGL perspective matrix from field of view in x and y axis

Args:
    fov_x (float | torch.Tensor): field of view in x axis
    fov_y (float | torch.Tensor): field of view in y axis
    near (float | torch.Tensor): near plane to clip
    far (float | torch.Tensor): far plane to clip

Returns:
    (torch.Tensor): [..., 4, 4] perspective matrix"""
    utils3d.torch.transforms.perspective_from_fov_xy

@overload
def intrinsics_from_focal_center(fx: Union[float, torch_.Tensor], fy: Union[float, torch_.Tensor], cx: Union[float, torch_.Tensor], cy: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenCV intrinsics matrix

Args:
    focal_x (float | torch.Tensor): focal length in x axis
    focal_y (float | torch.Tensor): focal length in y axis
    cx (float | torch.Tensor): principal point in x axis
    cy (float | torch.Tensor): principal point in y axis

Returns:
    (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.torch.transforms.intrinsics_from_focal_center

@overload
def intrinsics_from_fov(fov_max: Union[float, torch_.Tensor] = None, fov_min: Union[float, torch_.Tensor] = None, fov_x: Union[float, torch_.Tensor] = None, fov_y: Union[float, torch_.Tensor] = None, width: Union[int, torch_.Tensor] = None, height: Union[int, torch_.Tensor] = None) -> torch_.Tensor:
    """Get normalized OpenCV intrinsics matrix from given field of view.
You can provide either fov_max, fov_min, fov_x or fov_y

Args:
    width (int | torch.Tensor): image width
    height (int | torch.Tensor): image height
    fov_max (float | torch.Tensor): field of view in largest dimension
    fov_min (float | torch.Tensor): field of view in smallest dimension
    fov_x (float | torch.Tensor): field of view in x axis
    fov_y (float | torch.Tensor): field of view in y axis

Returns:
    (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.torch.transforms.intrinsics_from_fov

@overload
def intrinsics_from_fov_xy(fov_x: Union[float, torch_.Tensor], fov_y: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenCV intrinsics matrix from field of view in x and y axis

Args:
    fov_x (float | torch.Tensor): field of view in x axis
    fov_y (float | torch.Tensor): field of view in y axis

Returns:
    (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.torch.transforms.intrinsics_from_fov_xy

@overload
def view_look_at(eye: torch_.Tensor, look_at: torch_.Tensor, up: torch_.Tensor) -> torch_.Tensor:
    """Get OpenGL view matrix looking at something

Args:
    eye (torch.Tensor): [..., 3] the eye position
    look_at (torch.Tensor): [..., 3] the position to look at
    up (torch.Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

Returns:
    (torch.Tensor): [..., 4, 4], view matrix"""
    utils3d.torch.transforms.view_look_at

@overload
def extrinsics_look_at(eye: torch_.Tensor, look_at: torch_.Tensor, up: torch_.Tensor) -> torch_.Tensor:
    """Get OpenCV extrinsics matrix looking at something

Args:
    eye (torch.Tensor): [..., 3] the eye position
    look_at (torch.Tensor): [..., 3] the position to look at
    up (torch.Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

Returns:
    (torch.Tensor): [..., 4, 4], extrinsics matrix"""
    utils3d.torch.transforms.extrinsics_look_at

@overload
def perspective_to_intrinsics(perspective: torch_.Tensor) -> torch_.Tensor:
    """OpenGL perspective matrix to OpenCV intrinsics

Args:
    perspective (torch.Tensor): [..., 4, 4] OpenGL perspective matrix

Returns:
    (torch.Tensor): shape [..., 3, 3] OpenCV intrinsics"""
    utils3d.torch.transforms.perspective_to_intrinsics

@overload
def intrinsics_to_perspective(intrinsics: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix

Args:
    intrinsics (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    near (float | torch.Tensor): [...] near plane to clip
    far (float | torch.Tensor): [...] far plane to clip
Returns:
    (torch.Tensor): [..., 4, 4] OpenGL perspective matrix"""
    utils3d.torch.transforms.intrinsics_to_perspective

@overload
def extrinsics_to_view(extrinsics: torch_.Tensor) -> torch_.Tensor:
    """OpenCV camera extrinsics to OpenGL view matrix

Args:
    extrinsics (torch.Tensor): [..., 4, 4] OpenCV camera extrinsics matrix

Returns:
    (torch.Tensor): [..., 4, 4] OpenGL view matrix"""
    utils3d.torch.transforms.extrinsics_to_view

@overload
def view_to_extrinsics(view: torch_.Tensor) -> torch_.Tensor:
    """OpenGL view matrix to OpenCV camera extrinsics

Args:
    view (torch.Tensor): [..., 4, 4] OpenGL view matrix

Returns:
    (torch.Tensor): [..., 4, 4] OpenCV camera extrinsics matrix"""
    utils3d.torch.transforms.view_to_extrinsics

@overload
def normalize_intrinsics(intrinsics: torch_.Tensor, width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor]) -> torch_.Tensor:
    """Normalize camera intrinsics(s) to uv space

Args:
    intrinsics (torch.Tensor): [..., 3, 3] camera intrinsics(s) to normalize
    width (int | torch.Tensor): [...] image width(s)
    height (int | torch.Tensor): [...] image height(s)

Returns:
    (torch.Tensor): [..., 3, 3] normalized camera intrinsics(s)"""
    utils3d.torch.transforms.normalize_intrinsics

@overload
def crop_intrinsics(intrinsics: torch_.Tensor, width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor], left: Union[int, torch_.Tensor], top: Union[int, torch_.Tensor], crop_width: Union[int, torch_.Tensor], crop_height: Union[int, torch_.Tensor]) -> torch_.Tensor:
    """Evaluate the new intrinsics(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

Args:
    intrinsics (torch.Tensor): [..., 3, 3] camera intrinsics(s) to crop
    width (int | torch.Tensor): [...] image width(s)
    height (int | torch.Tensor): [...] image height(s)
    left (int | torch.Tensor): [...] left crop boundary
    top (int | torch.Tensor): [...] top crop boundary
    crop_width (int | torch.Tensor): [...] crop width
    crop_height (int | torch.Tensor): [...] crop height

Returns:
    (torch.Tensor): [..., 3, 3] cropped camera intrinsics(s)"""
    utils3d.torch.transforms.crop_intrinsics

@overload
def pixel_to_uv(pixel: torch_.Tensor, width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor]) -> torch_.Tensor:
    """Args:
    pixel (torch.Tensor): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
    width (int | torch.Tensor): [...] image width(s)
    height (int | torch.Tensor): [...] image height(s)

Returns:
    (torch.Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.torch.transforms.pixel_to_uv

@overload
def pixel_to_ndc(pixel: torch_.Tensor, width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor]) -> torch_.Tensor:
    """Args:
    pixel (torch.Tensor): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
    width (int | torch.Tensor): [...] image width(s)
    height (int | torch.Tensor): [...] image height(s)

Returns:
    (torch.Tensor): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)"""
    utils3d.torch.transforms.pixel_to_ndc

@overload
def uv_to_pixel(uv: torch_.Tensor, width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor]) -> torch_.Tensor:
    """Args:
    uv (torch.Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    width (int | torch.Tensor): [...] image width(s)
    height (int | torch.Tensor): [...] image height(s)

Returns:
    (torch.Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.torch.transforms.uv_to_pixel

@overload
def project_depth(depth: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Project linear depth to depth value in screen space

Args:
    depth (torch.Tensor): [...] depth value
    near (float | torch.Tensor): [...] near plane to clip
    far (float | torch.Tensor): [...] far plane to clip

Returns:
    (torch.Tensor): [..., 1] depth value in screen space, value ranging in [0, 1]"""
    utils3d.torch.transforms.project_depth

@overload
def depth_buffer_to_linear(depth: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Linearize depth value to linear depth

Args:
    depth (torch.Tensor): [...] screen depth value, ranging in [0, 1]
    near (float | torch.Tensor): [...] near plane to clip
    far (float | torch.Tensor): [...] far plane to clip

Returns:
    (torch.Tensor): [...] linear depth"""
    utils3d.torch.transforms.depth_buffer_to_linear

@overload
def project_gl(points: torch_.Tensor, model: torch_.Tensor = None, view: torch_.Tensor = None, perspective: torch_.Tensor = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrice)

Args:
    points (torch.Tensor): [..., N, 3 or 4] 3D points to project, if the last 
        dimension is 4, the points are assumed to be in homogeneous coordinates
    model (torch.Tensor): [..., 4, 4] model matrix
    view (torch.Tensor): [..., 4, 4] view matrix
    perspective (torch.Tensor): [..., 4, 4] perspective matrix

Returns:
    scr_coord (torch.Tensor): [..., N, 3] screen space coordinates, value ranging in [0, 1].
        The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
    linear_depth (torch.Tensor): [..., N] linear depth"""
    utils3d.torch.transforms.project_gl

@overload
def project_cv(points: torch_.Tensor, extrinsics: torch_.Tensor = None, intrinsics: torch_.Tensor = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Project 3D points to 2D following the OpenCV convention

Args:
    points (torch.Tensor): [..., N, 3] or [..., N, 4] 3D points to project, if the last
        dimension is 4, the points are assumed to be in homogeneous coordinates
    extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix
    intrinsics (torch.Tensor): [..., 3, 3] intrinsics matrix

Returns:
    uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    linear_depth (torch.Tensor): [..., N] linear depth"""
    utils3d.torch.transforms.project_cv

@overload
def unproject_gl(screen_coord: torch_.Tensor, model: torch_.Tensor = None, view: torch_.Tensor = None, perspective: torch_.Tensor = None) -> torch_.Tensor:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

Args:
    screen_coord (torch.Tensor): [... N, 3] screen space coordinates, value ranging in [0, 1].
        The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
    model (torch.Tensor): [..., 4, 4] model matrix
    view (torch.Tensor): [..., 4, 4] view matrix
    perspective (torch.Tensor): [..., 4, 4] perspective matrix

Returns:
    points (torch.Tensor): [..., N, 3] 3d points"""
    utils3d.torch.transforms.unproject_gl

@overload
def unproject_cv(uv_coord: torch_.Tensor, depth: torch_.Tensor, extrinsics: torch_.Tensor = None, intrinsics: torch_.Tensor = None) -> torch_.Tensor:
    """Unproject uv coordinates to 3D view space following the OpenCV convention

Args:
    uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    depth (torch.Tensor): [..., N] depth value
    extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix
    intrinsics (torch.Tensor): [..., 3, 3] intrinsics matrix

Returns:
    points (torch.Tensor): [..., N, 3] 3d points"""
    utils3d.torch.transforms.unproject_cv

@overload
def skew_symmetric(v: torch_.Tensor):
    """Skew symmetric matrix from a 3D vector"""
    utils3d.torch.transforms.skew_symmetric

@overload
def rotation_matrix_from_vectors(v1: torch_.Tensor, v2: torch_.Tensor):
    """Rotation matrix that rotates v1 to v2"""
    utils3d.torch.transforms.rotation_matrix_from_vectors

@overload
def euler_axis_angle_rotation(axis: str, angle: torch_.Tensor) -> torch_.Tensor:
    """Return the rotation matrices for one of the rotations about an axis
of which Euler angles describe, for each value of the angle given.

Args:
    axis: Axis label "X" or "Y or "Z".
    angle: any shape tensor of Euler angles in radians

Returns:
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.torch.transforms.euler_axis_angle_rotation

@overload
def euler_angles_to_matrix(euler_angles: torch_.Tensor, convention: str = 'XYZ') -> torch_.Tensor:
    """Convert rotations given as Euler angles in radians to rotation matrices.

Args:
    euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
    convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

Returns:
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.torch.transforms.euler_angles_to_matrix

@overload
def matrix_to_euler_angles(matrix: torch_.Tensor, convention: str) -> torch_.Tensor:
    """Convert rotations given as rotation matrices to Euler angles in radians.
NOTE: The composition order eg. `XYZ` means `Rz * Ry * Rx` (like blender), instead of `Rx * Ry * Rz` (like pytorch3d)

Args:
    matrix: Rotation matrices as tensor of shape (..., 3, 3).
    convention: Convention string of three uppercase letters.

Returns:
    Euler angles in radians as tensor of shape (..., 3), in the order of XYZ (like blender), instead of convention (like pytorch3d)"""
    utils3d.torch.transforms.matrix_to_euler_angles

@overload
def matrix_to_quaternion(rot_mat: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

Args:
    rot_mat (torch.Tensor): shape (..., 3, 3), the rotation matrices to convert

Returns:
    torch.Tensor: shape (..., 4), the quaternions corresponding to the given rotation matrices"""
    utils3d.torch.transforms.matrix_to_quaternion

@overload
def quaternion_to_matrix(quaternion: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices

Args:
    quaternion (torch.Tensor): shape (..., 4), the quaternions to convert

Returns:
    torch.Tensor: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions"""
    utils3d.torch.transforms.quaternion_to_matrix

@overload
def matrix_to_axis_angle(rot_mat: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

Args:
    rot_mat (torch.Tensor): shape (..., 3, 3), the rotation matrices to convert

Returns:
    torch.Tensor: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices"""
    utils3d.torch.transforms.matrix_to_axis_angle

@overload
def axis_angle_to_matrix(axis_angle: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

Args:
    axis_angle (torch.Tensor): shape (..., 3), axis-angle vcetors

Returns:
    torch.Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters"""
    utils3d.torch.transforms.axis_angle_to_matrix

@overload
def axis_angle_to_quaternion(axis_angle: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

Args:
    axis_angle (torch.Tensor): shape (..., 3), axis-angle vcetors

Returns:
    torch.Tensor: shape (..., 4) The quaternions for the given axis-angle parameters"""
    utils3d.torch.transforms.axis_angle_to_quaternion

@overload
def quaternion_to_axis_angle(quaternion: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

Args:
    quaternion (torch.Tensor): shape (..., 4), the quaternions to convert

Returns:
    torch.Tensor: shape (..., 3), the axis-angle vectors corresponding to the given quaternions"""
    utils3d.torch.transforms.quaternion_to_axis_angle

@overload
def slerp(rot_mat_1: torch_.Tensor, rot_mat_2: torch_.Tensor, t: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Spherical linear interpolation between two rotation matrices

Args:
    rot_mat_1 (torch.Tensor): shape (..., 3, 3), the first rotation matrix
    rot_mat_2 (torch.Tensor): shape (..., 3, 3), the second rotation matrix
    t (torch.Tensor): scalar or shape (...,), the interpolation factor

Returns:
    torch.Tensor: shape (..., 3, 3), the interpolated rotation matrix"""
    utils3d.torch.transforms.slerp

@overload
def interpolate_extrinsics(ext1: torch_.Tensor, ext2: torch_.Tensor, t: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Interpolate extrinsics between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

Args:
    ext1 (torch.Tensor): shape (..., 4, 4), the first camera pose
    ext2 (torch.Tensor): shape (..., 4, 4), the second camera pose
    t (torch.Tensor): scalar or shape (...,), the interpolation factor

Returns:
    torch.Tensor: shape (..., 4, 4), the interpolated camera pose"""
    utils3d.torch.transforms.interpolate_extrinsics

@overload
def interpolate_view(view1: torch_.Tensor, view2: torch_.Tensor, t: Union[numbers.Number, torch_.Tensor]):
    """Interpolate view matrices between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

Args:
    ext1 (torch.Tensor): shape (..., 4, 4), the first camera pose
    ext2 (torch.Tensor): shape (..., 4, 4), the second camera pose
    t (torch.Tensor): scalar or shape (...,), the interpolation factor

Returns:
    torch.Tensor: shape (..., 4, 4), the interpolated camera pose"""
    utils3d.torch.transforms.interpolate_view

@overload
def extrinsics_to_essential(extrinsics: torch_.Tensor):
    """extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

Args:
    extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix

Returns:
    (torch.Tensor): [..., 3, 3] essential matrix"""
    utils3d.torch.transforms.extrinsics_to_essential

@overload
def to4x4(R: torch_.Tensor, t: torch_.Tensor):
    """Compose rotation matrix and translation vector to 4x4 transformation matrix

Args:
    R (torch.Tensor): [..., 3, 3] rotation matrix
    t (torch.Tensor): [..., 3] translation vector

Returns:
    (torch.Tensor): [..., 4, 4] transformation matrix"""
    utils3d.torch.transforms.to4x4

@overload
def rotation_matrix_2d(theta: Union[float, torch_.Tensor]):
    """2x2 matrix for 2D rotation

Args:
    theta (float | torch.Tensor): rotation angle in radians, arbitrary shape (...,)

Returns:
    (torch.Tensor): (..., 2, 2) rotation matrix"""
    utils3d.torch.transforms.rotation_matrix_2d

@overload
def rotate_2d(theta: Union[float, torch_.Tensor], center: torch_.Tensor = None):
    """3x3 matrix for 2D rotation around a center
```
   [[Rxx, Rxy, tx],
    [Ryx, Ryy, ty],
    [0,     0,  1]]
```
Args:
    theta (float | torch.Tensor): rotation angle in radians, arbitrary shape (...,)
    center (torch.Tensor): rotation center, arbitrary shape (..., 2). Default to (0, 0)
    
Returns:
    (torch.Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.rotate_2d

@overload
def translate_2d(translation: torch_.Tensor):
    """Translation matrix for 2D translation
```
   [[1, 0, tx],
    [0, 1, ty],
    [0, 0,  1]]
```
Args:
    translation (torch.Tensor): translation vector, arbitrary shape (..., 2)

Returns:
    (torch.Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.translate_2d

@overload
def scale_2d(scale: Union[float, torch_.Tensor], center: torch_.Tensor = None):
    """Scale matrix for 2D scaling
```
   [[s, 0, tx],
    [0, s, ty],
    [0, 0,  1]]
```
Args:
    scale (float | torch.Tensor): scale factor, arbitrary shape (...,)
    center (torch.Tensor): scale center, arbitrary shape (..., 2). Default to (0, 0)

Returns:
    (torch.Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.scale_2d

@overload
def apply_2d(transform: torch_.Tensor, points: torch_.Tensor):
    """Apply (3x3 or 2x3) 2D affine transformation to points
```
    p = R @ p + t
```
Args:
    transform (torch.Tensor): (..., 2 or 3, 3) transformation matrix
    points (torch.Tensor): (..., N, 2) points to transform

Returns:
    (torch.Tensor): (..., N, 2) transformed points"""
    utils3d.torch.transforms.apply_2d

@overload
def RastContext(nvd_ctx: Union[nvdiffrast.torch.ops.RasterizeCudaContext, nvdiffrast.torch.ops.RasterizeGLContext] = None, *, backend: Literal['cuda', 'gl'] = 'gl', device: Union[str, torch_.device] = None):
    """Create a rasterization context. Nothing but a wrapper of nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext."""
    utils3d.torch.rasterization.RastContext

@overload
def rasterize_triangle_faces(ctx: utils3d.torch.rasterization.RastContext, vertices: torch_.Tensor, faces: torch_.Tensor, attr: torch_.Tensor, width: int, height: int, model: torch_.Tensor = None, view: torch_.Tensor = None, projection: torch_.Tensor = None, antialiasing: Union[bool, List[int]] = True, diff_attrs: Optional[List[int]] = None) -> Tuple[torch_.Tensor, torch_.Tensor, Optional[torch_.Tensor]]:
    """Rasterize a mesh with vertex attributes.

Args:
    ctx (GLContext): rasterizer context
    vertices (np.ndarray): (B, N, 2 or 3 or 4)
    faces (torch.Tensor): (T, 3)
    attr (torch.Tensor): (B, N, C)
    width (int): width of the output image
    height (int): height of the output image
    model (torch.Tensor, optional): ([B,] 4, 4) model matrix. Defaults to None (identity).
    view (torch.Tensor, optional): ([B,] 4, 4) view matrix. Defaults to None (identity).
    projection (torch.Tensor, optional): ([B,] 4, 4) projection matrix. Defaults to None (identity).
    antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
    diff_attrs (Union[None, List[int]], optional): indices of attributes to compute screen-space derivatives. Defaults to None.

Returns:
    image: (torch.Tensor): (B, C, H, W)
    depth: (torch.Tensor): (B, H, W) screen space depth, ranging from 0 (near) to 1. (far)
        NOTE: Empty pixels will have depth 1., i.e. far plane."""
    utils3d.torch.rasterization.rasterize_triangle_faces

@overload
def warp_image_by_depth(ctx: utils3d.torch.rasterization.RastContext, depth: torch_.FloatTensor, image: torch_.FloatTensor = None, mask: torch_.BoolTensor = None, width: int = None, height: int = None, *, extrinsics_src: torch_.FloatTensor = None, extrinsics_tgt: torch_.FloatTensor = None, intrinsics_src: torch_.FloatTensor = None, intrinsics_tgt: torch_.FloatTensor = None, near: float = 0.1, far: float = 100.0, antialiasing: bool = True, backslash: bool = False, padding: int = 0, return_uv: bool = False, return_dr: bool = False) -> Tuple[torch_.FloatTensor, torch_.FloatTensor, torch_.BoolTensor, Optional[torch_.FloatTensor], Optional[torch_.FloatTensor]]:
    """Warp image by depth. 
NOTE: if batch size is 1, image mesh will be triangulated aware of the depth, yielding less distorted results.
Otherwise, image mesh will be triangulated simply for batch rendering.

Args:
    ctx (Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]): rasterization context
    depth (torch.Tensor): (B, H, W) linear depth
    image (torch.Tensor): (B, C, H, W). None to use image space uv. Defaults to None.
    width (int, optional): width of the output image. None to use the same as depth. Defaults to None.
    height (int, optional): height of the output image. Defaults the same as depth..
    extrinsics_src (torch.Tensor, optional): (B, 4, 4) extrinsics matrix for source. None to use identity. Defaults to None.
    extrinsics_tgt (torch.Tensor, optional): (B, 4, 4) extrinsics matrix for target. None to use identity. Defaults to None.
    intrinsics_src (torch.Tensor, optional): (B, 3, 3) intrinsics matrix for source. None to use the same as target. Defaults to None.
    intrinsics_tgt (torch.Tensor, optional): (B, 3, 3) intrinsics matrix for target. None to use the same as source. Defaults to None.
    near (float, optional): near plane. Defaults to 0.1. 
    far (float, optional): far plane. Defaults to 100.0.
    antialiasing (bool, optional): whether to perform antialiasing. Defaults to True.
    backslash (bool, optional): whether to use backslash triangulation. Defaults to False.
    padding (int, optional): padding of the image. Defaults to 0.
    return_uv (bool, optional): whether to return the uv. Defaults to False.
    return_dr (bool, optional): whether to return the image-space derivatives of uv. Defaults to False.

Returns:
    image: (torch.FloatTensor): (B, C, H, W) rendered image
    depth: (torch.FloatTensor): (B, H, W) linear depth, ranging from 0 to inf
    mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
    uv: (torch.FloatTensor): (B, 2, H, W) image-space uv
    dr: (torch.FloatTensor): (B, 4, H, W) image-space derivatives of uv"""
    utils3d.torch.rasterization.warp_image_by_depth

@overload
def warp_image_by_forward_flow(ctx: utils3d.torch.rasterization.RastContext, image: torch_.FloatTensor, flow: torch_.FloatTensor, depth: torch_.FloatTensor = None, *, antialiasing: bool = True, backslash: bool = False) -> Tuple[torch_.FloatTensor, torch_.BoolTensor]:
    """Warp image by forward flow.
NOTE: if batch size is 1, image mesh will be triangulated aware of the depth, yielding less distorted results.
Otherwise, image mesh will be triangulated simply for batch rendering.

Args:
    ctx (Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]): rasterization context
    image (torch.Tensor): (B, C, H, W) image
    flow (torch.Tensor): (B, 2, H, W) forward flow
    depth (torch.Tensor, optional): (B, H, W) linear depth. If None, will use the same for all pixels. Defaults to None.
    antialiasing (bool, optional): whether to perform antialiasing. Defaults to True.
    backslash (bool, optional): whether to use backslash triangulation. Defaults to False.

Returns:
    image: (torch.FloatTensor): (B, C, H, W) rendered image
    mask: (torch.BoolTensor): (B, H, W) mask of valid pixels"""
    utils3d.torch.rasterization.warp_image_by_forward_flow

