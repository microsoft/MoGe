import importlib
import itertools
import torch
from typing import TYPE_CHECKING

__modules_all__ = {
    'mesh': [
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
    ],
    'nerf': [
        'get_rays',
        'get_image_rays',
        'get_mipnerf_cones',
        'volume_rendering',
        'bin_sample',
        'importance_sample',
        'nerf_render_rays',
        'mipnerf_render_rays',
        'nerf_render_view',
        'mipnerf_render_view',
        'InstantNGP',
    ],
    'utils': [
        'sliding_window_1d',
        'sliding_window_2d',
        'sliding_window_nd',
        'image_uv',
        'image_pixel_center',
        'image_mesh',
        'chessboard',
        'depth_edge',
        'depth_aliasing',
        'image_mesh_from_depth',
        'point_to_normal',
        'depth_to_normal',
        'masked_min',
        'masked_max',
        'bounding_rect'
    ],
    'transforms': [
        'perspective',
        'perspective_from_fov',
        'perspective_from_fov_xy',
        'intrinsics_from_focal_center',
        'intrinsics_from_fov',
        'intrinsics_from_fov_xy',
        'view_look_at',
        'extrinsics_look_at',
        'perspective_to_intrinsics',
        'intrinsics_to_perspective',
        'extrinsics_to_view',
        'view_to_extrinsics',
        'normalize_intrinsics',
        'crop_intrinsics',
        'pixel_to_uv',
        'pixel_to_ndc',
        'uv_to_pixel',
        'project_depth',
        'depth_buffer_to_linear',
        'project_gl',
        'project_cv',
        'unproject_gl',
        'unproject_cv',
        'skew_symmetric',
        'rotation_matrix_from_vectors',
        'euler_axis_angle_rotation',
        'euler_angles_to_matrix',
        'matrix_to_euler_angles',
        'matrix_to_quaternion',
        'quaternion_to_matrix',
        'matrix_to_axis_angle',
        'axis_angle_to_matrix',
        'axis_angle_to_quaternion',
        'quaternion_to_axis_angle',
        'slerp',
        'interpolate_extrinsics',
        'interpolate_view',
        'extrinsics_to_essential',
        'to4x4',
        'rotation_matrix_2d',
        'rotate_2d',
        'translate_2d',
        'scale_2d',
        'apply_2d',
    ],
    'rasterization': [
        'RastContext',
        'rasterize_triangle_faces', 
        'warp_image_by_depth',
        'warp_image_by_forward_flow',
    ],
}


__all__ = list(itertools.chain(*__modules_all__.values()))

def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass

    try:
        module_name = next(m for m in __modules_all__ if name in __modules_all__[m])
    except StopIteration:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(f'.{module_name}', __name__)
    for key in __modules_all__[module_name]:
        globals()[key] = getattr(module, key)
        
    return globals()[name]


if TYPE_CHECKING:
    from .transforms import *
    from .mesh import *
    from .utils import *
    from .nerf import *
    from .rasterization import *