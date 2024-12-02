# Auto-generated implementation redirecting to numpy/torch implementations
import sys
from typing import TYPE_CHECKING
import utils3d
from .._helpers import suppress_traceback

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

def _contains_tensor(obj):
    if isinstance(obj, (list, tuple)):
        return any(_contains_tensor(item) for item in obj)
    elif isinstance(obj, dict):
        return any(_contains_tensor(value) for value in obj.values())
    else:
        import torch
        return isinstance(obj, torch.Tensor)


@suppress_traceback
def _call_based_on_args(fname, args, kwargs):
    if 'torch' in sys.modules:
        if any(_contains_tensor(arg) for arg in args) or any(_contains_tensor(v) for v in kwargs.values()):
            fn = getattr(utils3d.torch, fname, None)
            if fn is None:
                raise NotImplementedError(f"Function {fname} has no torch implementation.")
            return fn(*args, **kwargs)
    fn = getattr(utils3d.numpy, fname, None)
    if fn is None:
        raise NotImplementedError(f"Function {fname} has no numpy implementation.") 
    return fn(*args, **kwargs)


@suppress_traceback
def triangulate(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.triangulate, utils3d.torch.triangulate
    return _call_based_on_args('triangulate', args, kwargs)

@suppress_traceback
def compute_face_normal(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_normal, utils3d.torch.compute_face_normal
    return _call_based_on_args('compute_face_normal', args, kwargs)

@suppress_traceback
def compute_face_angle(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_angle, None
    return _call_based_on_args('compute_face_angle', args, kwargs)

@suppress_traceback
def compute_vertex_normal(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_vertex_normal, utils3d.torch.compute_vertex_normal
    return _call_based_on_args('compute_vertex_normal', args, kwargs)

@suppress_traceback
def compute_vertex_normal_weighted(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_vertex_normal_weighted, utils3d.torch.compute_vertex_normal_weighted
    return _call_based_on_args('compute_vertex_normal_weighted', args, kwargs)

@suppress_traceback
def remove_corrupted_faces(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.remove_corrupted_faces, utils3d.torch.remove_corrupted_faces
    return _call_based_on_args('remove_corrupted_faces', args, kwargs)

@suppress_traceback
def merge_duplicate_vertices(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.merge_duplicate_vertices, utils3d.torch.merge_duplicate_vertices
    return _call_based_on_args('merge_duplicate_vertices', args, kwargs)

@suppress_traceback
def remove_unreferenced_vertices(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.remove_unreferenced_vertices, utils3d.torch.remove_unreferenced_vertices
    return _call_based_on_args('remove_unreferenced_vertices', args, kwargs)

@suppress_traceback
def subdivide_mesh_simple(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.subdivide_mesh_simple, utils3d.torch.subdivide_mesh_simple
    return _call_based_on_args('subdivide_mesh_simple', args, kwargs)

@suppress_traceback
def mesh_relations(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_relations, None
    return _call_based_on_args('mesh_relations', args, kwargs)

@suppress_traceback
def flatten_mesh_indices(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.flatten_mesh_indices, None
    return _call_based_on_args('flatten_mesh_indices', args, kwargs)

@suppress_traceback
def calc_quad_candidates(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.calc_quad_candidates, None
    return _call_based_on_args('calc_quad_candidates', args, kwargs)

@suppress_traceback
def calc_quad_distortion(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.calc_quad_distortion, None
    return _call_based_on_args('calc_quad_distortion', args, kwargs)

@suppress_traceback
def calc_quad_direction(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.calc_quad_direction, None
    return _call_based_on_args('calc_quad_direction', args, kwargs)

@suppress_traceback
def calc_quad_smoothness(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.calc_quad_smoothness, None
    return _call_based_on_args('calc_quad_smoothness', args, kwargs)

@suppress_traceback
def sovle_quad(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sovle_quad, None
    return _call_based_on_args('sovle_quad', args, kwargs)

@suppress_traceback
def sovle_quad_qp(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sovle_quad_qp, None
    return _call_based_on_args('sovle_quad_qp', args, kwargs)

@suppress_traceback
def tri_to_quad(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.tri_to_quad, None
    return _call_based_on_args('tri_to_quad', args, kwargs)

@suppress_traceback
def sliding_window_1d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sliding_window_1d, utils3d.torch.sliding_window_1d
    return _call_based_on_args('sliding_window_1d', args, kwargs)

@suppress_traceback
def sliding_window_nd(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sliding_window_nd, utils3d.torch.sliding_window_nd
    return _call_based_on_args('sliding_window_nd', args, kwargs)

@suppress_traceback
def sliding_window_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sliding_window_2d, utils3d.torch.sliding_window_2d
    return _call_based_on_args('sliding_window_2d', args, kwargs)

@suppress_traceback
def max_pool_1d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.max_pool_1d, None
    return _call_based_on_args('max_pool_1d', args, kwargs)

@suppress_traceback
def max_pool_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.max_pool_2d, None
    return _call_based_on_args('max_pool_2d', args, kwargs)

@suppress_traceback
def max_pool_nd(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.max_pool_nd, None
    return _call_based_on_args('max_pool_nd', args, kwargs)

@suppress_traceback
def depth_edge(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_edge, utils3d.torch.depth_edge
    return _call_based_on_args('depth_edge', args, kwargs)

@suppress_traceback
def normals_edge(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.normals_edge, None
    return _call_based_on_args('normals_edge', args, kwargs)

@suppress_traceback
def depth_aliasing(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_aliasing, utils3d.torch.depth_aliasing
    return _call_based_on_args('depth_aliasing', args, kwargs)

@suppress_traceback
def interpolate(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.interpolate, None
    return _call_based_on_args('interpolate', args, kwargs)

@suppress_traceback
def image_scrcoord(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.image_scrcoord, None
    return _call_based_on_args('image_scrcoord', args, kwargs)

@suppress_traceback
def image_uv(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.image_uv, utils3d.torch.image_uv
    return _call_based_on_args('image_uv', args, kwargs)

@suppress_traceback
def image_pixel_center(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.image_pixel_center, utils3d.torch.image_pixel_center
    return _call_based_on_args('image_pixel_center', args, kwargs)

@suppress_traceback
def image_pixel(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.image_pixel, None
    return _call_based_on_args('image_pixel', args, kwargs)

@suppress_traceback
def image_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.image_mesh, utils3d.torch.image_mesh
    return _call_based_on_args('image_mesh', args, kwargs)

@suppress_traceback
def image_mesh_from_depth(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.image_mesh_from_depth, utils3d.torch.image_mesh_from_depth
    return _call_based_on_args('image_mesh_from_depth', args, kwargs)

@suppress_traceback
def depth_to_normals(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_to_normals, None
    return _call_based_on_args('depth_to_normals', args, kwargs)

@suppress_traceback
def points_to_normals(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.points_to_normals, None
    return _call_based_on_args('points_to_normals', args, kwargs)

@suppress_traceback
def chessboard(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.chessboard, utils3d.torch.chessboard
    return _call_based_on_args('chessboard', args, kwargs)

@suppress_traceback
def cube(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.cube, None
    return _call_based_on_args('cube', args, kwargs)

@suppress_traceback
def icosahedron(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.icosahedron, None
    return _call_based_on_args('icosahedron', args, kwargs)

@suppress_traceback
def square(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.square, None
    return _call_based_on_args('square', args, kwargs)

@suppress_traceback
def camera_frustum(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.camera_frustum, None
    return _call_based_on_args('camera_frustum', args, kwargs)

@suppress_traceback
def perspective(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective, utils3d.torch.perspective
    return _call_based_on_args('perspective', args, kwargs)

@suppress_traceback
def perspective_from_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_from_fov, utils3d.torch.perspective_from_fov
    return _call_based_on_args('perspective_from_fov', args, kwargs)

@suppress_traceback
def perspective_from_fov_xy(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_from_fov_xy, utils3d.torch.perspective_from_fov_xy
    return _call_based_on_args('perspective_from_fov_xy', args, kwargs)

@suppress_traceback
def intrinsics_from_focal_center(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_from_focal_center, utils3d.torch.intrinsics_from_focal_center
    return _call_based_on_args('intrinsics_from_focal_center', args, kwargs)

@suppress_traceback
def intrinsics_from_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_from_fov, utils3d.torch.intrinsics_from_fov
    return _call_based_on_args('intrinsics_from_fov', args, kwargs)

@suppress_traceback
def fov_to_focal(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.fov_to_focal, None
    return _call_based_on_args('fov_to_focal', args, kwargs)

@suppress_traceback
def focal_to_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.focal_to_fov, None
    return _call_based_on_args('focal_to_fov', args, kwargs)

@suppress_traceback
def intrinsics_to_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_to_fov, None
    return _call_based_on_args('intrinsics_to_fov', args, kwargs)

@suppress_traceback
def view_look_at(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.view_look_at, utils3d.torch.view_look_at
    return _call_based_on_args('view_look_at', args, kwargs)

@suppress_traceback
def extrinsics_look_at(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_look_at, utils3d.torch.extrinsics_look_at
    return _call_based_on_args('extrinsics_look_at', args, kwargs)

@suppress_traceback
def perspective_to_intrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_to_intrinsics, utils3d.torch.perspective_to_intrinsics
    return _call_based_on_args('perspective_to_intrinsics', args, kwargs)

@suppress_traceback
def perspective_to_near_far(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_to_near_far, None
    return _call_based_on_args('perspective_to_near_far', args, kwargs)

@suppress_traceback
def intrinsics_to_perspective(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_to_perspective, utils3d.torch.intrinsics_to_perspective
    return _call_based_on_args('intrinsics_to_perspective', args, kwargs)

@suppress_traceback
def extrinsics_to_view(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_to_view, utils3d.torch.extrinsics_to_view
    return _call_based_on_args('extrinsics_to_view', args, kwargs)

@suppress_traceback
def view_to_extrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.view_to_extrinsics, utils3d.torch.view_to_extrinsics
    return _call_based_on_args('view_to_extrinsics', args, kwargs)

@suppress_traceback
def normalize_intrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.normalize_intrinsics, utils3d.torch.normalize_intrinsics
    return _call_based_on_args('normalize_intrinsics', args, kwargs)

@suppress_traceback
def crop_intrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.crop_intrinsics, utils3d.torch.crop_intrinsics
    return _call_based_on_args('crop_intrinsics', args, kwargs)

@suppress_traceback
def pixel_to_uv(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_to_uv, utils3d.torch.pixel_to_uv
    return _call_based_on_args('pixel_to_uv', args, kwargs)

@suppress_traceback
def pixel_to_ndc(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_to_ndc, utils3d.torch.pixel_to_ndc
    return _call_based_on_args('pixel_to_ndc', args, kwargs)

@suppress_traceback
def uv_to_pixel(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.uv_to_pixel, utils3d.torch.uv_to_pixel
    return _call_based_on_args('uv_to_pixel', args, kwargs)

@suppress_traceback
def project_depth(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project_depth, utils3d.torch.project_depth
    return _call_based_on_args('project_depth', args, kwargs)

@suppress_traceback
def depth_buffer_to_linear(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_buffer_to_linear, utils3d.torch.depth_buffer_to_linear
    return _call_based_on_args('depth_buffer_to_linear', args, kwargs)

@suppress_traceback
def unproject_cv(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject_cv, utils3d.torch.unproject_cv
    return _call_based_on_args('unproject_cv', args, kwargs)

@suppress_traceback
def unproject_gl(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject_gl, utils3d.torch.unproject_gl
    return _call_based_on_args('unproject_gl', args, kwargs)

@suppress_traceback
def project_cv(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project_cv, utils3d.torch.project_cv
    return _call_based_on_args('project_cv', args, kwargs)

@suppress_traceback
def project_gl(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project_gl, utils3d.torch.project_gl
    return _call_based_on_args('project_gl', args, kwargs)

@suppress_traceback
def quaternion_to_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_to_matrix, utils3d.torch.quaternion_to_matrix
    return _call_based_on_args('quaternion_to_matrix', args, kwargs)

@suppress_traceback
def axis_angle_to_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.axis_angle_to_matrix, utils3d.torch.axis_angle_to_matrix
    return _call_based_on_args('axis_angle_to_matrix', args, kwargs)

@suppress_traceback
def matrix_to_quaternion(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.matrix_to_quaternion, utils3d.torch.matrix_to_quaternion
    return _call_based_on_args('matrix_to_quaternion', args, kwargs)

@suppress_traceback
def extrinsics_to_essential(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_to_essential, utils3d.torch.extrinsics_to_essential
    return _call_based_on_args('extrinsics_to_essential', args, kwargs)

@suppress_traceback
def euler_axis_angle_rotation(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.euler_axis_angle_rotation, utils3d.torch.euler_axis_angle_rotation
    return _call_based_on_args('euler_axis_angle_rotation', args, kwargs)

@suppress_traceback
def euler_angles_to_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.euler_angles_to_matrix, utils3d.torch.euler_angles_to_matrix
    return _call_based_on_args('euler_angles_to_matrix', args, kwargs)

@suppress_traceback
def skew_symmetric(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.skew_symmetric, utils3d.torch.skew_symmetric
    return _call_based_on_args('skew_symmetric', args, kwargs)

@suppress_traceback
def rotation_matrix_from_vectors(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rotation_matrix_from_vectors, utils3d.torch.rotation_matrix_from_vectors
    return _call_based_on_args('rotation_matrix_from_vectors', args, kwargs)

@suppress_traceback
def ray_intersection(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.ray_intersection, None
    return _call_based_on_args('ray_intersection', args, kwargs)

@suppress_traceback
def se3_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.se3_matrix, None
    return _call_based_on_args('se3_matrix', args, kwargs)

@suppress_traceback
def slerp_quaternion(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.slerp_quaternion, None
    return _call_based_on_args('slerp_quaternion', args, kwargs)

@suppress_traceback
def slerp_vector(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.slerp_vector, None
    return _call_based_on_args('slerp_vector', args, kwargs)

@suppress_traceback
def lerp(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lerp, None
    return _call_based_on_args('lerp', args, kwargs)

@suppress_traceback
def lerp_se3_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lerp_se3_matrix, None
    return _call_based_on_args('lerp_se3_matrix', args, kwargs)

@suppress_traceback
def piecewise_lerp(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.piecewise_lerp, None
    return _call_based_on_args('piecewise_lerp', args, kwargs)

@suppress_traceback
def piecewise_lerp_se3_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.piecewise_lerp_se3_matrix, None
    return _call_based_on_args('piecewise_lerp_se3_matrix', args, kwargs)

@suppress_traceback
def apply_transform(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.apply_transform, None
    return _call_based_on_args('apply_transform', args, kwargs)

@suppress_traceback
def linear_spline_interpolate(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.linear_spline_interpolate, None
    return _call_based_on_args('linear_spline_interpolate', args, kwargs)

@suppress_traceback
def RastContext(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.RastContext, utils3d.torch.RastContext
    return _call_based_on_args('RastContext', args, kwargs)

@suppress_traceback
def rasterize_triangle_faces(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_triangle_faces, utils3d.torch.rasterize_triangle_faces
    return _call_based_on_args('rasterize_triangle_faces', args, kwargs)

@suppress_traceback
def rasterize_edges(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_edges, None
    return _call_based_on_args('rasterize_edges', args, kwargs)

@suppress_traceback
def texture(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.texture, None
    return _call_based_on_args('texture', args, kwargs)

@suppress_traceback
def warp_image_by_depth(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.warp_image_by_depth, utils3d.torch.warp_image_by_depth
    return _call_based_on_args('warp_image_by_depth', args, kwargs)

@suppress_traceback
def test_rasterization(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.test_rasterization, None
    return _call_based_on_args('test_rasterization', args, kwargs)

@suppress_traceback
def compute_face_angles(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.compute_face_angles
    return _call_based_on_args('compute_face_angles', args, kwargs)

@suppress_traceback
def compute_face_tbn(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.compute_face_tbn
    return _call_based_on_args('compute_face_tbn', args, kwargs)

@suppress_traceback
def compute_vertex_tbn(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.compute_vertex_tbn
    return _call_based_on_args('compute_vertex_tbn', args, kwargs)

@suppress_traceback
def laplacian(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.laplacian
    return _call_based_on_args('laplacian', args, kwargs)

@suppress_traceback
def laplacian_smooth_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.laplacian_smooth_mesh
    return _call_based_on_args('laplacian_smooth_mesh', args, kwargs)

@suppress_traceback
def taubin_smooth_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.taubin_smooth_mesh
    return _call_based_on_args('taubin_smooth_mesh', args, kwargs)

@suppress_traceback
def laplacian_hc_smooth_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.laplacian_hc_smooth_mesh
    return _call_based_on_args('laplacian_hc_smooth_mesh', args, kwargs)

@suppress_traceback
def get_rays(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.get_rays
    return _call_based_on_args('get_rays', args, kwargs)

@suppress_traceback
def get_image_rays(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.get_image_rays
    return _call_based_on_args('get_image_rays', args, kwargs)

@suppress_traceback
def get_mipnerf_cones(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.get_mipnerf_cones
    return _call_based_on_args('get_mipnerf_cones', args, kwargs)

@suppress_traceback
def volume_rendering(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.volume_rendering
    return _call_based_on_args('volume_rendering', args, kwargs)

@suppress_traceback
def bin_sample(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.bin_sample
    return _call_based_on_args('bin_sample', args, kwargs)

@suppress_traceback
def importance_sample(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.importance_sample
    return _call_based_on_args('importance_sample', args, kwargs)

@suppress_traceback
def nerf_render_rays(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.nerf_render_rays
    return _call_based_on_args('nerf_render_rays', args, kwargs)

@suppress_traceback
def mipnerf_render_rays(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.mipnerf_render_rays
    return _call_based_on_args('mipnerf_render_rays', args, kwargs)

@suppress_traceback
def nerf_render_view(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.nerf_render_view
    return _call_based_on_args('nerf_render_view', args, kwargs)

@suppress_traceback
def mipnerf_render_view(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.mipnerf_render_view
    return _call_based_on_args('mipnerf_render_view', args, kwargs)

@suppress_traceback
def InstantNGP(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.InstantNGP
    return _call_based_on_args('InstantNGP', args, kwargs)

@suppress_traceback
def point_to_normal(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.point_to_normal
    return _call_based_on_args('point_to_normal', args, kwargs)

@suppress_traceback
def depth_to_normal(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.depth_to_normal
    return _call_based_on_args('depth_to_normal', args, kwargs)

@suppress_traceback
def masked_min(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.masked_min
    return _call_based_on_args('masked_min', args, kwargs)

@suppress_traceback
def masked_max(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.masked_max
    return _call_based_on_args('masked_max', args, kwargs)

@suppress_traceback
def bounding_rect(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.bounding_rect
    return _call_based_on_args('bounding_rect', args, kwargs)

@suppress_traceback
def intrinsics_from_fov_xy(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.intrinsics_from_fov_xy
    return _call_based_on_args('intrinsics_from_fov_xy', args, kwargs)

@suppress_traceback
def matrix_to_euler_angles(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.matrix_to_euler_angles
    return _call_based_on_args('matrix_to_euler_angles', args, kwargs)

@suppress_traceback
def matrix_to_axis_angle(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.matrix_to_axis_angle
    return _call_based_on_args('matrix_to_axis_angle', args, kwargs)

@suppress_traceback
def axis_angle_to_quaternion(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.axis_angle_to_quaternion
    return _call_based_on_args('axis_angle_to_quaternion', args, kwargs)

@suppress_traceback
def quaternion_to_axis_angle(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.quaternion_to_axis_angle
    return _call_based_on_args('quaternion_to_axis_angle', args, kwargs)

@suppress_traceback
def slerp(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.slerp
    return _call_based_on_args('slerp', args, kwargs)

@suppress_traceback
def interpolate_extrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.interpolate_extrinsics
    return _call_based_on_args('interpolate_extrinsics', args, kwargs)

@suppress_traceback
def interpolate_view(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.interpolate_view
    return _call_based_on_args('interpolate_view', args, kwargs)

@suppress_traceback
def to4x4(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.to4x4
    return _call_based_on_args('to4x4', args, kwargs)

@suppress_traceback
def rotation_matrix_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.rotation_matrix_2d
    return _call_based_on_args('rotation_matrix_2d', args, kwargs)

@suppress_traceback
def rotate_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.rotate_2d
    return _call_based_on_args('rotate_2d', args, kwargs)

@suppress_traceback
def translate_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.translate_2d
    return _call_based_on_args('translate_2d', args, kwargs)

@suppress_traceback
def scale_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.scale_2d
    return _call_based_on_args('scale_2d', args, kwargs)

@suppress_traceback
def apply_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.apply_2d
    return _call_based_on_args('apply_2d', args, kwargs)

@suppress_traceback
def warp_image_by_forward_flow(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.warp_image_by_forward_flow
    return _call_based_on_args('warp_image_by_forward_flow', args, kwargs)

