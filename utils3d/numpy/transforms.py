import numpy as np
from typing import *
from numbers import Number
from ._helpers import batched
from .._helpers import no_warnings


__all__ = [
    'perspective',
    'perspective_from_fov',
    'perspective_from_fov_xy',
    'intrinsics_from_focal_center',
    'intrinsics_from_fov',
    'fov_to_focal',
    'focal_to_fov',
    'intrinsics_to_fov',
    'view_look_at',
    'extrinsics_look_at',
    'perspective_to_intrinsics',
    'perspective_to_near_far',
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
    'unproject_cv',
    'unproject_gl',
    'project_cv',
    'project_gl',
    'quaternion_to_matrix',
    'axis_angle_to_matrix',
    'matrix_to_quaternion',
    'extrinsics_to_essential',
    'euler_axis_angle_rotation',
    'euler_angles_to_matrix',
    'skew_symmetric',
    'rotation_matrix_from_vectors',
    'ray_intersection',
    'se3_matrix',
    'slerp_quaternion',
    'slerp_vector',
    'lerp',
    'lerp_se3_matrix',
    'piecewise_lerp',
    'piecewise_lerp_se3_matrix',
    'apply_transform'
]


@batched(0,0,0,0)
def perspective(
    fov_y: Union[float, np.ndarray],
    aspect: Union[float, np.ndarray],
    near: Union[float, np.ndarray],
    far: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Get OpenGL perspective matrix

    Args:
        fov_y (float | np.ndarray): field of view in y axis
        aspect (float | np.ndarray): aspect ratio
        near (float | np.ndarray): near plane to clip
        far (float | np.ndarray): far plane to clip

    Returns:
        (np.ndarray): [..., 4, 4] perspective matrix
    """
    N = fov_y.shape[0]
    ret = np.zeros((N, 4, 4), dtype=fov_y.dtype)
    ret[:, 0, 0] = 1. / (np.tan(fov_y / 2) * aspect)
    ret[:, 1, 1] = 1. / (np.tan(fov_y / 2))
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret


def perspective_from_fov(
    fov: Union[float, np.ndarray],
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray],
    near: Union[float, np.ndarray],
    far: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Get OpenGL perspective matrix from field of view in largest dimension

    Args:
        fov (float | np.ndarray): field of view in largest dimension
        width (int | np.ndarray): image width
        height (int | np.ndarray): image height
        near (float | np.ndarray): near plane to clip
        far (float | np.ndarray): far plane to clip

    Returns:
        (np.ndarray): [..., 4, 4] perspective matrix
    """
    fov_y = 2 * np.arctan(np.tan(fov / 2) * height / np.maximum(width, height))
    aspect = width / height
    return perspective(fov_y, aspect, near, far)


def perspective_from_fov_xy(
    fov_x: Union[float, np.ndarray],
    fov_y: Union[float, np.ndarray],
    near: Union[float, np.ndarray],
    far: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Get OpenGL perspective matrix from field of view in x and y axis

    Args:
        fov_x (float | np.ndarray): field of view in x axis
        fov_y (float | np.ndarray): field of view in y axis
        near (float | np.ndarray): near plane to clip
        far (float | np.ndarray): far plane to clip

    Returns:
        (np.ndarray): [..., 4, 4] perspective matrix
    """
    aspect = np.tan(fov_x / 2) / np.tan(fov_y / 2)
    return perspective(fov_y, aspect, near, far)


def intrinsics_from_focal_center(
    fx: Union[float, np.ndarray],
    fy: Union[float, np.ndarray],
    cx: Union[float, np.ndarray],
    cy: Union[float, np.ndarray],
    dtype: Optional[np.dtype] = np.float32
) -> np.ndarray:
    """
    Get OpenCV intrinsics matrix

    Returns:
        (np.ndarray): [..., 3, 3] OpenCV intrinsics matrix
    """
    if any(isinstance(x, np.ndarray) for x in (fx, fy, cx, cy)):
        dtype = np.result_type(fx, fy, cx, cy)
    fx, fy, cx, cy = np.broadcast_arrays(fx, fy, cx, cy)
    ret = np.zeros((*fx.shape, 3, 3), dtype=dtype)
    ret[..., 0, 0] = fx
    ret[..., 1, 1] = fy
    ret[..., 0, 2] = cx
    ret[..., 1, 2] = cy
    ret[..., 2, 2] = 1.
    return ret


def intrinsics_from_fov(
    fov_max: Union[float, np.ndarray] = None,
    fov_min: Union[float, np.ndarray] = None,
    fov_x: Union[float, np.ndarray] = None,
    fov_y: Union[float, np.ndarray] = None,
    width: Union[int, np.ndarray] = None,
    height: Union[int, np.ndarray] = None,
) -> np.ndarray:
    """
    Get normalized OpenCV intrinsics matrix from given field of view.
    You can provide either fov_max, fov_min, fov_x or fov_y

    Args:
        width (int | np.ndarray): image width
        height (int | np.ndarray): image height
        fov_max (float | np.ndarray): field of view in largest dimension
        fov_min (float | np.ndarray): field of view in smallest dimension
        fov_x (float | np.ndarray): field of view in x axis
        fov_y (float | np.ndarray): field of view in y axis

    Returns:
        (np.ndarray): [..., 3, 3] OpenCV intrinsics matrix
    """
    if fov_max is not None:
        fx = np.maximum(width, height) / width / (2 * np.tan(fov_max / 2))
        fy = np.maximum(width, height) / height / (2 * np.tan(fov_max / 2))
    elif fov_min is not None:
        fx = np.minimum(width, height) / width / (2 * np.tan(fov_min / 2))
        fy = np.minimum(width, height) / height / (2 * np.tan(fov_min / 2))
    elif fov_x is not None and fov_y is not None:
        fx = 1 / (2 * np.tan(fov_x / 2))
        fy = 1 / (2 * np.tan(fov_y / 2))
    elif fov_x is not None:
        fx = 1 / (2 * np.tan(fov_x / 2))
        fy = fx * width / height
    elif fov_y is not None:
        fy = 1 / (2 * np.tan(fov_y / 2))
        fx = fy * height / width
    cx = 0.5
    cy = 0.5
    ret = intrinsics_from_focal_center(fx, fy, cx, cy)
    return ret


def focal_to_fov(focal: np.ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal(fov: np.ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov(intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fov_x = focal_to_fov(intrinsics[..., 0, 0])
    fov_y = focal_to_fov(intrinsics[..., 1, 1])
    return fov_x, fov_y


@batched(1,1,1)
def view_look_at(
        eye: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray
    ) -> np.ndarray:
    """
    Get OpenGL view matrix looking at something

    Args:
        eye (np.ndarray): [..., 3] the eye position
        look_at (np.ndarray): [..., 3] the position to look at
        up (np.ndarray): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (np.ndarray): [..., 4, 4], view matrix
    """
    z = eye - look_at
    x = np.cross(up, z)
    y = np.cross(z, x)
    # x = np.cross(y, z)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    R = np.stack([x, y, z], axis=-2)
    t = -np.matmul(R, eye[..., None])
    return np.concatenate([
        np.concatenate([R, t], axis=-1),
        np.array([[[0., 0., 0., 1.]]]).repeat(eye.shape[0], axis=0)
    ], axis=-2)


@batched(1,1,1)
def extrinsics_look_at(
    eye: np.ndarray,
    look_at: np.ndarray,
    up: np.ndarray
) -> np.ndarray:
    """
    Get OpenCV extrinsics matrix looking at something

    Args:
        eye (np.ndarray): [..., 3] the eye position
        look_at (np.ndarray): [..., 3] the position to look at
        up (np.ndarray): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (np.ndarray): [..., 4, 4], extrinsics matrix
    """
    z = look_at - eye
    x = np.cross(-up, z)
    y = np.cross(z, x)
    # x = np.cross(y, z)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    R = np.stack([x, y, z], axis=-2)
    t = -np.matmul(R, eye[..., None])
    return np.concatenate([
        np.concatenate([R, t], axis=-1),
        np.array([[[0., 0., 0., 1.]]], dtype=eye.dtype).repeat(eye.shape[0], axis=0)
    ], axis=-2)


def perspective_to_intrinsics(
    perspective: np.ndarray
) -> np.ndarray:
    """
    OpenGL perspective matrix to OpenCV intrinsics

    Args:
        perspective (np.ndarray): [..., 4, 4] OpenGL perspective matrix

    Returns:
        (np.ndarray): shape [..., 3, 3] OpenCV intrinsics
    """
    ret = np.array([[0.5, 0., 0.5], [0., -0.5, 0.5], [0., 0., 1.]], dtype=perspective.dtype) \
        @ perspective[..., [0, 1, 3], :3] \
        @ np.diag(np.array([1, -1, -1], dtype=perspective.dtype))
    return ret


def perspective_to_near_far(perspective: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get near and far planes from OpenGL perspective matrix

    Args:
    """
    a, b = perspective[..., 2, 2], perspective[..., 2, 3]
    near, far =  b / (a - 1), b / (a + 1)
    return near, far


@batched(2,0,0)
def intrinsics_to_perspective(
    intrinsics: np.ndarray,
    near: Union[float, np.ndarray],
    far: Union[float, np.ndarray],
) -> np.ndarray:
    """
    OpenCV intrinsics to OpenGL perspective matrix
    NOTE: not work for tile-shifting intrinsics currently

    Args:
        intrinsics (np.ndarray): [..., 3, 3] OpenCV intrinsics matrix
        near (float | np.ndarray): [...] near plane to clip
        far (float | np.ndarray): [...] far plane to clip
    Returns:
        (np.ndarray): [..., 4, 4] OpenGL perspective matrix
    """
    N = intrinsics.shape[0]
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    ret = np.zeros((N, 4, 4), dtype=intrinsics.dtype)
    ret[:, 0, 0] = 2 * fx
    ret[:, 1, 1] = 2 * fy
    ret[:, 0, 2] = -2 * cx + 1
    ret[:, 1, 2] = 2 * cy - 1
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret


@batched(2)
def extrinsics_to_view(
        extrinsics: np.ndarray
    ) -> np.ndarray:
    """
    OpenCV camera extrinsics to OpenGL view matrix

    Args:
        extrinsics (np.ndarray): [..., 4, 4] OpenCV camera extrinsics matrix

    Returns:
        (np.ndarray): [..., 4, 4] OpenGL view matrix
    """
    return extrinsics * np.array([1, -1, -1, 1], dtype=extrinsics.dtype)[:, None]


@batched(2)
def view_to_extrinsics(
        view: np.ndarray
    ) -> np.ndarray:
    """
    OpenGL view matrix to OpenCV camera extrinsics

    Args:
        view (np.ndarray): [..., 4, 4] OpenGL view matrix

    Returns:
        (np.ndarray): [..., 4, 4] OpenCV camera extrinsics matrix
    """
    return view * np.array([1, -1, -1, 1], dtype=view.dtype)[:, None]


@batched(2, 0, 0, None)
def normalize_intrinsics(
    intrinsics: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray],
    integer_pixel_centers: bool = True
) -> np.ndarray:
    """
    Normalize intrinsics from pixel cooridnates to uv coordinates

    Args:
        intrinsics (np.ndarray): [..., 3, 3] camera intrinsics(s) to normalize
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)
        integer_pixel_centers (bool): whether the integer pixel coordinates are at the center of the pixel. If False, the integer coordinates are at the left-top corner of the pixel.

    Returns:
        (np.ndarray): [..., 3, 3] normalized camera intrinsics(s)
    """
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    if integer_pixel_centers:
        transform = np.stack([
            1 / width, zeros, 0.5 / width,
            zeros, 1 / height, 0.5 / height,
            zeros, zeros, ones
        ]).reshape(*zeros.shape, 3, 3)
    else:
        transform = np.stack([
            1 / width, zeros, zeros,
            zeros, 1 / height, zeros,
            zeros, zeros, ones
        ]).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@batched(2,0,0,0,0,0,0)
def crop_intrinsics(
    intrinsics: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray],
    left: Union[int, np.ndarray],
    top: Union[int, np.ndarray],
    crop_width: Union[int, np.ndarray],
    crop_height: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Evaluate the new intrinsics(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

    Args:
        intrinsics (np.ndarray): [..., 3, 3] camera intrinsics(s) to crop
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)
        left (int | np.ndarray): [...] left crop boundary
        top (int | np.ndarray): [...] top crop boundary
        crop_width (int | np.ndarray): [...] crop width
        crop_height (int | np.ndarray): [...] crop height

    Returns:
        (np.ndarray): [..., 3, 3] cropped camera intrinsics(s)
    """
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    transform = np.stack([
        width / crop_width, zeros, -left / crop_width,
        zeros, height / crop_height, -top / crop_height,
        zeros, zeros, ones
    ]).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@batched(1,0,0)
def pixel_to_uv(
    pixel: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    if not np.issubdtype(pixel.dtype, np.floating):
        pixel = pixel.astype(np.float32)
    dtype = pixel.dtype
    uv = (pixel + np.array(0.5, dtype=dtype)) / np.stack([width, height], axis=-1)
    return uv


@batched(1,0,0)
def uv_to_pixel(
    uv: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    pixel = uv * np.stack([width, height], axis=-1).astype(uv.dtype) - 0.5
    return pixel


@batched(1,0,0)
def pixel_to_ndc(
    pixel: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)
    """
    if not np.issubdtype(pixel.dtype, np.floating):
        pixel = pixel.astype(np.float32)
    dtype = pixel.dtype
    ndc = (pixel + np.array(0.5, dtype=dtype)) / (np.stack([width, height], dim=-1) * np.array([2, -2], dtype=dtype)) \
        + np.array([-1, 1], dtype=dtype)
    return ndc


@batched(0,0,0)
def project_depth(
    depth: np.ndarray,
    near: Union[float, np.ndarray],
    far: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Project linear depth to depth value in screen space

    Args:
        depth (np.ndarray): [...] depth value
        near (float | np.ndarray): [...] near plane to clip
        far (float | np.ndarray): [...] far plane to clip

    Returns:
        (np.ndarray): [..., 1] depth value in screen space, value ranging in [0, 1]
    """
    return (far - near * far / depth) / (far - near)


@batched(0,0,0)
def depth_buffer_to_linear(
    depth_buffer: np.ndarray,
    near: Union[float, np.ndarray],
    far: Union[float, np.ndarray]
) -> np.ndarray:
    """
    OpenGL depth buffer to linear depth

    Args:
        depth_buffer (np.ndarray): [...] depth value
        near (float | np.ndarray): [...] near plane to clip
        far (float | np.ndarray): [...] far plane to clip

    Returns:
        (np.ndarray): [..., 1] linear depth
    """
    return near * far / (far - (far - near) * depth_buffer)


@batched(2,2,2,2)
def project_gl(
        points: np.ndarray,
        model: np.ndarray = None,
        view: np.ndarray = None,
        perspective: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        points (np.ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
            dimension is 4, the points are assumed to be in homogeneous coordinates
        model (np.ndarray): [..., 4, 4] model matrix
        view (np.ndarray): [..., 4, 4] view matrix
        perspective (np.ndarray): [..., 4, 4] perspective matrix

    Returns:
        scr_coord (np.ndarray): [..., N, 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        linear_depth (np.ndarray): [..., N] linear depth
    """
    assert perspective is not None, "perspective matrix is required"
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    if model is not None:
        points = points @ model.swapaxes(-1, -2)
    if view is not None:
        points = points @ view.swapaxes(-1, -2)
    clip_coord = points @ perspective.swapaxes(-1, -2)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth


@batched(2,2,2)
def project_cv(
        points: np.ndarray,
        extrinsics: np.ndarray = None,
        intrinsics: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D following the OpenCV convention

    Args:
        points (np.ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last
            dimension is 4, the points are assumed to be in homogeneous coordinates
        extrinsics (np.ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (np.ndarray): [..., 3, 3] intrinsics matrix

    Returns:
        uv_coord (np.ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (np.ndarray): [..., N] linear depth
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    if extrinsics is not None:
        points = points @ extrinsics.swapaxes(-1, -2)
    points = points[..., :3] @ intrinsics.swapaxes(-1, -2)
    with no_warnings():
        uv_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


@batched(2,2,2,2)
def unproject_gl(
        screen_coord: np.ndarray,
        model: np.ndarray = None,
        view: np.ndarray = None,
        perspective: np.ndarray = None
    ) -> np.ndarray:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

    Args:
        screen_coord (np.ndarray): [..., N, 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        model (np.ndarray): [..., 4, 4] model matrix
        view (np.ndarray): [..., 4, 4] view matrix
        perspective (np.ndarray): [..., 4, 4] perspective matrix

    Returns:
        points (np.ndarray): [..., N, 3] 3d points
    """
    assert perspective is not None, "perspective matrix is required"
    ndc_xy = screen_coord * 2 - 1
    clip_coord = np.concatenate([ndc_xy, np.ones_like(ndc_xy[..., :1])], axis=-1)
    transform = perspective
    if view is not None:
        transform = transform @ view
    if model is not None:
        transform = transform @ model
    transform = np.linalg.inv(transform)
    points = clip_coord @ transform.swapaxes(-1, -2)
    points = points[..., :3] / points[..., 3:]
    return points
    

@batched(2,1,2,2)
def unproject_cv(
    uv_coord: np.ndarray,
    depth: np.ndarray = None,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None
) -> np.ndarray:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (np.ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (np.ndarray): [..., N] depth value
        extrinsics (np.ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (np.ndarray): [..., 3, 3] intrinsics matrix

    Returns:
        points (np.ndarray): [..., N, 3] 3d points
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = np.concatenate([uv_coord, np.ones_like(uv_coord[..., :1])], axis=-1)
    points = points @ np.linalg.inv(intrinsics).swapaxes(-1, -2)
    if depth is not None:
        points = points * depth[..., None]
    if extrinsics is not None:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        points = (points @ np.linalg.inv(extrinsics).swapaxes(-1, -2))[..., :3]
    return points


def quaternion_to_matrix(quaternion: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices
    
    Args:
        quaternion (np.ndarray): shape (..., 4), the quaternions to convert
    
    Returns:
        np.ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=eps)
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    zeros = np.zeros_like(w)
    I = np.eye(3, dtype=quaternion.dtype)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * (xyz ** 2).sum(axis=-1)[..., None, None]
    B = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], axis=-1).reshape(*quaternion.shape[:-1], 3, 3)
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def matrix_to_quaternion(rot_mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    Args:
        rot_mat (np.ndarray): shape (..., 3, 3), the rotation matrices to convert

    Returns:
        np.ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [rot_mat[..., i, j] for i in range(3) for j in range(3)]

    diag = np.diagonal(rot_mat, axis1=-2, axis2=-1)
    M = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype)
    wxyz = 0.5 * np.clip(1 + diag @ M.T, 0.0, None) ** 0.5
    max_idx = np.argmax(wxyz, axis=-1)
    xw = np.sign(m21 - m12)
    yw = np.sign(m02 - m20)
    zw = np.sign(m10 - m01)
    yz = np.sign(m21 + m12)
    xz = np.sign(m02 + m20)
    xy = np.sign(m01 + m10)
    ones = np.ones_like(xw)
    sign = np.where(
        max_idx[..., None] == 0,
        np.stack([ones, xw, yw, zw], axis=-1),
        np.where(
            max_idx[..., None] == 1,
            np.stack([xw, ones, xy, xz], axis=-1),
            np.where(
                max_idx[..., None] == 2,
                np.stack([yw, xy, ones, yz], axis=-1),
                np.stack([zw, xz, yz, ones], axis=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=eps)
    return quat


def extrinsics_to_essential(extrinsics: np.ndarray):
    """
    extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

    Args:
        extrinsics (np.ndaray): [..., 4, 4] extrinsics matrix

    Returns:
        (np.ndaray): [..., 3, 3] essential matrix
    """
    assert extrinsics.shape[-2:] == (4, 4)
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    zeros = np.zeros_like(t[..., 0])
    t_x = np.stack([
        zeros, -t[..., 2], t[..., 1],
        t[..., 2], zeros, -t[..., 0],
        -t[..., 1], t[..., 0], zeros
    ]).reshape(*t.shape[:-1], 3, 3)
    return t_x @ R 


def euler_axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: np.ndarray, convention: str = 'XYZ') -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as ndarray of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    Returns:
        Rotation matrices as ndarray of shape (..., 3, 3).
    """
    if euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        euler_axis_angle_rotation(c, euler_angles[..., 'XYZ'.index(c)])
        for c in convention
    ]
    return matrices[2] @ matrices[1] @ matrices[0]


def skew_symmetric(v: np.ndarray):
    "Skew symmetric matrix from a 3D vector"
    assert v.shape[-1] == 3, "v must be 3D"
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = np.zeros_like(x)
    return np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], axis=-1).reshape(*v.shape[:-1], 3, 3)


def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray):
    "Rotation matrix that rotates v1 to v2"
    I = np.eye(3, dtype=v1.dtype)
    v1 = v1 / np.linalg.norm(v1, axis=-1)
    v2 = v2 / np.linalg.norm(v2, axis=-1)
    v = np.cross(v1, v2, axis=-1)
    c = np.sum(v1 * v2, axis=-1)
    K = skew_symmetric(v)
    R = I + K + (1 / (1 + c)).astype(v1.dtype)[None, None] * (K @ K)    # Avoid numpy's default type casting for scalars
    return R


def axis_angle_to_matrix(axis_angle: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

    Args:
        axis_angle (np.ndarray): shape (..., 3), axis-angle vcetors

    Returns:
        np.ndarray: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = axis_angle.shape[:-1]
    dtype = axis_angle.dtype

    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True) 
    axis = axis_angle / (angle + eps)

    cos = np.cos(angle)[..., None, :]
    sin = np.sin(angle)[..., None, :]

    rx, ry, rz = np.split(axis, 3, axis=-1)
    zeros = np.zeros((*batch_shape, 1), dtype=dtype)
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=-1).reshape((*batch_shape, 3, 3))

    ident = np.eye(3, dtype=dtype)
    rot_mat = ident + sin * K + (1 - cos) * (K @ K)
    return rot_mat


def ray_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray):
    """
    Compute the intersection/closest point of two D-dimensional rays
    If the rays are intersecting, the closest point is the intersection point.

    Args:
        p1 (np.ndarray): (..., D) origin of ray 1
        d1 (np.ndarray): (..., D) direction of ray 1
        p2 (np.ndarray): (..., D) origin of ray 2
        d2 (np.ndarray): (..., D) direction of ray 2

    Returns:
        (np.ndarray): (..., N) intersection point
    """
    p1, d1, p2, d2 = np.broadcast_arrays(p1, d1, p2, d2)
    dtype = p1.dtype
    dim = p1.shape[-1]
    d = np.stack([d1, d2], axis=-2)     # (..., 2, D)
    p = np.stack([p1, p2], axis=-2)     # (..., 2, D)
    A = np.concatenate([
        (np.eye(dim, dtype=dtype) * np.ones((*p.shape[:-2], 2, 1, 1))).reshape(*d.shape[:-2], 2 * dim, dim),         # (..., 2 * D, D)
        -(np.eye(2, dtype=dtype)[..., None] * d[..., None, :]).swapaxes(-2, -1).reshape(*d.shape[:-2], 2 * dim, 2)    # (..., 2 * D, 2)
    ], axis=-1)                             # (..., 2 * D, D + 2)
    b = p.reshape(*p.shape[:-2], 2 * dim)   # (..., 2 * D)
    x = np.linalg.solve(A.swapaxes(-1, -2) @ A + 1e-12 * np.eye(dim + 2, dtype=dtype), (A.swapaxes(-1, -2) @ b[..., :, None])[..., 0])
    return x[..., :dim], (x[..., dim], x[..., dim + 1])


def se3_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix and translation vector to 4x4 transformation matrix.

    Args:
        R (np.ndarray): [..., 3, 3] rotation matrix
        t (np.ndarray): [..., 3] translation vector

    Returns:
        np.ndarray: [..., 4, 4] transformation matrix
    """
    assert R.shape[:-2] == t.shape[:-1]
    assert R.shape[-1] == 3 and R.shape[-2] == 3
    return np.concatenate([
        np.concatenate([R, t[..., None]], axis=-1), 
        np.concatenate([np.zeros_like(t), np.ones_like(t[..., :1])], axis=-1)[..., None, :]
    ], axis=-2)


def slerp_quaternion(q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Spherical linear interpolation between two unit quaternions.

    Args:
        q1 (np.ndarray): [..., d] unit vector 1
        q2 (np.ndarray): [..., d] unit vector 2
        t (np.ndarray): [...] interpolation parameter in [0, 1]

    Returns:
        np.ndarray: [..., 3] interpolated unit vector
    """
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)

    dot = np.where(dot < 0, -dot, dot)  # handle negative dot product

    dot = np.minimum(dot, 1.)
    theta = np.arccos(dot) * t

    q_ortho = q2 - q1 * dot
    q_ortho = q_ortho / np.maximum(np.linalg.norm(q_ortho, axis=-1, keepdims=True), 1e-12)
    q = q1 * np.cos(theta) + q_ortho * np.sin(theta)
    return q


def slerp_rotation_matrix(R1: np.ndarray, R2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Spherical linear interpolation between two rotation matrices.

    Args:
        R1 (np.ndarray): [..., 3, 3] rotation matrix 1
        R2 (np.ndarray): [..., 3, 3] rotation matrix 2
        t (np.ndarray): [...] interpolation parameter in [0, 1]

    Returns:
        np.ndarray: [..., 3, 3] interpolated rotation matrix
    """
    quat1 = matrix_to_quaternion(R1)
    quat2 = matrix_to_quaternion(R2)
    quat = slerp_quaternion(quat1, quat2, t)
    return quaternion_to_matrix(quat)


def slerp_vector(v1: np.ndarray, v2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Spherical linear interpolation between two unit vectors. The vectors are assumed to be normalized.

    Args:
        v1 (np.ndarray): [..., d] unit vector 1
        v2 (np.ndarray): [..., d] unit vector 2
        t (np.ndarray): [...] interpolation parameter in [0, 1]

    Returns:
        np.ndarray: [..., d] interpolated unit vector
    """
    dot = np.sum(v1 * v2, axis=-1, keepdims=True)

    dot = np.minimum(dot, 1.)
    theta = np.arccos(dot) * t

    v_ortho = v2 - v1 * dot
    v_ortho = v_ortho / np.maximum(np.linalg.norm(v_ortho, axis=-1, keepdims=True), 1e-12)
    v = v1 * np.cos(theta) + v_ortho * np.sin(theta)
    return v


def lerp(x1: np.ndarray, x2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Linear interpolation between two vectors.

    Args:
        x1 (np.ndarray): [..., d] vector 1
        x2 (np.ndarray): [..., d] vector 2
        t (np.ndarray): [...] interpolation parameter. [0, 1] for interpolation between x1 and x2, otherwise for extrapolation.

    Returns:
        np.ndarray: [..., d] interpolated vector
    """
    return x1 + np.asarray(t)[..., None] * (x2 - x1)


def lerp_se3_matrix(T1: np.ndarray, T2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Linear interpolation between two SE(3) matrices.

    Args:
        T1 (np.ndarray): [..., 4, 4] SE(3) matrix 1
        T2 (np.ndarray): [..., 4, 4] SE(3) matrix 2
        t (np.ndarray): [...] interpolation parameter in [0, 1]

    Returns:
        np.ndarray: [..., 4, 4] interpolated SE(3) matrix
    """
    R1 = T1[..., :3, :3]
    R2 = T2[..., :3, :3]
    trans1 = T1[..., :3, 3]
    trans2 = T2[..., :3, 3]
    R = slerp_rotation_matrix(R1, R2, t)
    trans = lerp(trans1, trans2, t)
    return se3_matrix(R, trans)


def piecewise_lerp(x: np.ndarray, t: np.ndarray, s: np.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> np.ndarray:
    """
    Linear spline interpolation.

    ### Parameters:
    - `x`: np.ndarray, shape (n, d): the values of data points.
    - `t`: np.ndarray, shape (n,): the times of the data points.
    - `s`: np.ndarray, shape (m,): the times to be interpolated.
    - `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.
    
    ### Returns:
    - `y`: np.ndarray, shape (..., m, d): the interpolated values.
    """
    i = np.searchsorted(t, s, side='left')
    if extrapolation_mode == 'constant':
        prev = np.clip(i - 1, 0, len(t) - 1)
        suc = np.clip(i, 0, len(t) - 1)
    elif extrapolation_mode == 'linear':
        prev = np.clip(i - 1, 0, len(t) - 2)
        suc = np.clip(i, 1, len(t) - 1)
    else:
        raise ValueError(f'Invalid extrapolation_mode: {extrapolation_mode}')
    
    u = (s - t[prev]) / np.maximum(t[suc] - t[prev], 1e-12)
    y = lerp(x[prev], x[suc], u)

    return y


def piecewise_lerp_se3_matrix(T: np.ndarray, t: np.ndarray, s: np.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> np.ndarray:
    """
    Linear spline interpolation for SE(3) matrices.

    ### Parameters:
    - `T`: np.ndarray, shape (n, 4, 4): the SE(3) matrices.
    - `t`: np.ndarray, shape (n,): the times of the data points.
    - `s`: np.ndarray, shape (m,): the times to be interpolated.
    - `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

    ### Returns:
    - `T_interp`: np.ndarray, shape (..., m, 4, 4): the interpolated SE(3) matrices.
    """
    i = np.searchsorted(t, s, side='left')
    if extrapolation_mode == 'constant':
        prev = np.clip(i - 1, 0, len(t) - 1)
        suc = np.clip(i, 0, len(t) - 1)
    elif extrapolation_mode == 'linear':
        prev = np.clip(i - 1, 0, len(t) - 2)
        suc = np.clip(i, 1, len(t) - 1)
    else:
        raise ValueError(f'Invalid extrapolation_mode: {extrapolation_mode}')
    
    u = (s - t[prev]) / np.maximum(t[suc] - t[prev], 1e-12)
    T = lerp_se3_matrix(T[prev], T[suc], u)

    return T


def apply_transform(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply SE(3) transformation to a point or a set of points.

    ### Parameters:
    - `T`: np.ndarray, shape (..., 4, 4): the SE(3) matrix.
    - `x`: np.ndarray, shape (..., 3): the point or a set of points to be transformed.

    ### Returns:
    - `x_transformed`: np.ndarray, shape (..., 3): the transformed point or a set of points.
    """
    x = np.asarray(x)
    assert x.shape[-1] == 3
    T = np.asarray(T)
    assert T.shape[-2:] == (4, 4)
    x_transformed = (T[..., :3, :3] @ x[..., :, None]) + T[..., :3, 3][..., None]
    return x_transformed[..., 0]