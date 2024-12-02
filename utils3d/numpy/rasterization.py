import os
from typing import *

import numpy as np
import moderngl

from . import transforms, utils, mesh


__all__ = [
    'RastContext',
    'rasterize_triangle_faces',
    'rasterize_edges',
    'texture',
    'test_rasterization',
    'warp_image_by_depth',
]


def map_np_dtype(dtype) -> str:
    if dtype == int:
        return 'i4'
    elif dtype == np.uint8:
        return 'u1'
    elif dtype == np.uint32:
        return 'u2'
    elif dtype == np.float16:
        return 'f2'
    elif dtype == np.float32:
        return 'f4'
    

def one_value(dtype):
    if dtype == 'u1':
        return 255
    elif dtype == 'u2':
        return 65535
    else:
        return 1


class RastContext:
    def __init__(self, *args, **kwargs):
        """
        Create a moderngl context.

        Args:
            See moderngl.create_context
        """
        if len(args) == 1 and isinstance(args[0], moderngl.Context):
            self.mgl_ctx = args[0]
        else:
            self.mgl_ctx = moderngl.create_context(*args, **kwargs)
        self.__prog_src = {}
        self.__prog = {}

    def program_vertex_attribute(self, n: int) -> moderngl.Program:
        assert n in [1, 2, 3, 4], 'vertex attribute only supports channels 1, 2, 3, 4'

        if 'vertex_attribute_vsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'vertex_attribute.vsh'), 'r') as f:
                self.__prog_src['vertex_attribute_vsh'] = f.read()
        if 'vertex_attribute_fsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'vertex_attribute.fsh'), 'r') as f:
                self.__prog_src['vertex_attribute_fsh'] = f.read()
        
        if f'vertex_attribute_{n}' not in self.__prog:
            vsh = self.__prog_src['vertex_attribute_vsh'].replace('vecN', f'vec{n}')
            fsh = self.__prog_src['vertex_attribute_fsh'].replace('vecN', f'vec{n}')
            self.__prog[f'vertex_attribute_{n}'] = self.mgl_ctx.program(vertex_shader=vsh, fragment_shader=fsh)

        return self.__prog[f'vertex_attribute_{n}']

    def program_texture(self, n: int) -> moderngl.Program:
        assert n in [1, 2, 3, 4], 'texture only supports channels 1, 2, 3, 4'

        if 'texture_vsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'texture.vsh'), 'r') as f:
                self.__prog_src['texture_vsh'] = f.read()
        if 'texture_fsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'texture.fsh'), 'r') as f:
                self.__prog_src['texture_fsh'] = f.read()

        if f'texture_{n}' not in self.__prog:
            vsh = self.__prog_src['texture_vsh'].replace('vecN', f'vec{n}')
            fsh = self.__prog_src['texture_fsh'].replace('vecN', f'vec{n}')
            self.__prog[f'texture_{n}'] = self.mgl_ctx.program(vertex_shader=vsh, fragment_shader=fsh)
            self.__prog[f'texture_{n}']['tex'] = 0
            self.__prog[f'texture_{n}']['uv'] = 1
        
        return self.__prog[f'texture_{n}']

    
def rasterize_triangle_faces(
    ctx: RastContext,
    vertices: np.ndarray,
    faces: np.ndarray,
    attr: np.ndarray,
    width: int,
    height: int,
    transform: np.ndarray = None,
    cull_backface: bool = True,
    return_depth: bool = False,
    image: np.ndarray = None,
    depth: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize vertex attribute.

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
        depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3, f"Faces should be a 2D array with shape (T, 3), but got {faces.shape}"
    assert attr.ndim == 2 and attr.shape[1] in [1, 2, 3, 4], f'Vertex attribute only supports channels 1, 2, 3, 4, but got {attr.shape}'
    assert vertices.shape[0] == attr.shape[0]
    assert vertices.dtype == np.float32
    assert faces.dtype == np.uint32 or faces.dtype == np.int32
    assert attr.dtype == np.float32, "Attribute should be float32"
    assert transform is None or transform.shape == (4, 4), f"Transform should be a 4x4 matrix, but got {transform.shape}"
    assert transform is None or transform.dtype == np.float32, f"Transform should be float32, but got {transform.dtype}"
    if image is not None:
        assert image.ndim == 3 and image.shape == (height, width, attr.shape[1]), f"Image should be a 3D array with shape (H, W, {attr.shape[1]}), but got {image.shape}"
        assert image.dtype == np.float32, f"Image should be float32, but got {image.dtype}"
    if depth is not None:
        assert depth.ndim == 2 and depth.shape == (height, width), f"Depth should be a 2D array with shape (H, W), but got {depth.shape}"
        assert depth.dtype == np.float32, f"Depth should be float32, but got {depth.dtype}"

    C = attr.shape[1]
    prog = ctx.program_vertex_attribute(C)

    transform = np.eye(4, np.float32) if transform is None else transform

    # Create buffers
    ibo = ctx.mgl_ctx.buffer(np.ascontiguousarray(faces, dtype='i4'))
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    vbo_attr = ctx.mgl_ctx.buffer(np.ascontiguousarray(attr, dtype='f4'))
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        [
            (vbo_vertices, '3f', 'i_position'),
            (vbo_attr, f'{C}f', 'i_attr'),
        ],
        ibo,
        mode=moderngl.TRIANGLES,
    )

    # Create framebuffer
    image_tex = ctx.mgl_ctx.texture((width, height), C, dtype='f4', data=np.ascontiguousarray(image[::-1, :, :]) if image is not None else None)
    depth_tex = ctx.mgl_ctx.depth_texture((width, height), data=np.ascontiguousarray(depth[::-1, :]) if depth is not None else None)
    fbo = ctx.mgl_ctx.framebuffer(
        color_attachments=[image_tex],
        depth_attachment=depth_tex,
    )

    # Render
    prog['u_mvp'].write(transform.transpose().copy().astype('f4'))
    fbo.use()
    fbo.viewport = (0, 0, width, height)
    ctx.mgl_ctx.depth_func = '<'
    if depth is None:
        ctx.mgl_ctx.clear(depth=1.0)
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    if cull_backface:
        ctx.mgl_ctx.enable(ctx.mgl_ctx.CULL_FACE)
    else:
        ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)
    vao.render()
    ctx.mgl_ctx.disable(ctx.mgl_ctx.DEPTH_TEST)

    # Read
    image = np.zeros((height, width, C), dtype='f4') 
    image_tex.read_into(image)
    image = image[::-1, :, :]
    if return_depth:
        depth = np.zeros((height, width), dtype='f4')
        depth_tex.read_into(depth)
        depth = depth[::-1, :]
    else:
        depth = None

    # Release
    vao.release()
    ibo.release()
    vbo_vertices.release()
    vbo_attr.release()
    fbo.release()
    image_tex.release()
    depth_tex.release()

    return image, depth


def rasterize_edges(
    ctx: RastContext,
    vertices: np.ndarray,
    edges: np.ndarray,
    attr: np.ndarray,
    width: int,
    height: int,
    transform: np.ndarray = None,
    line_width: float = 1.0,
    return_depth: bool = False,
    image: np.ndarray = None,
    depth: np.ndarray = None
) -> Tuple[np.ndarray, ...]:
    """
    Rasterize vertex attribute.

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
        depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert edges.ndim == 2 and edges.shape[1] == 2, f"Edges should be a 2D array with shape (T, 2), but got {edges.shape}"
    assert attr.ndim == 2 and attr.shape[1] in [1, 2, 3, 4], f'Vertex attribute only supports channels 1, 2, 3, 4, but got {attr.shape}'
    assert vertices.shape[0] == attr.shape[0]
    assert vertices.dtype == np.float32
    assert edges.dtype == np.uint32 or edges.dtype == np.int32
    assert attr.dtype == np.float32, "Attribute should be float32"

    C = attr.shape[1]
    prog = ctx.program_vertex_attribute(C)

    transform = transform if transform is not None else np.eye(4, np.float32)

    # Create buffers
    ibo = ctx.mgl_ctx.buffer(np.ascontiguousarray(edges, dtype='i4'))
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    vbo_attr = ctx.mgl_ctx.buffer(np.ascontiguousarray(attr, dtype='f4'))
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        [
            (vbo_vertices, '3f', 'i_position'),
            (vbo_attr, f'{C}f', 'i_attr'),
        ],
        ibo,
        mode=moderngl.LINES,
    )

    # Create framebuffer
    image_tex = ctx.mgl_ctx.texture((width, height), C, dtype='f4', data=np.ascontiguousarray(image[::-1, :, :]) if image is not None else None)
    depth_tex = ctx.mgl_ctx.depth_texture((width, height), data=np.ascontiguousarray(depth[::-1, :]) if depth is not None else None)
    fbo = ctx.mgl_ctx.framebuffer(
        color_attachments=[image_tex],
        depth_attachment=depth_tex,
    )

    # Render
    prog['u_mvp'].write(transform.transpose().copy().astype('f4'))
    fbo.use()
    fbo.viewport = (0, 0, width, height)
    if depth is None:
        ctx.mgl_ctx.clear(depth=1.0)
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    ctx.mgl_ctx.line_width = line_width
    vao.render()
    ctx.mgl_ctx.disable(ctx.mgl_ctx.DEPTH_TEST)

    # Read
    image = np.zeros((height, width, C), dtype='f4')
    image_tex.read_into(image)
    image = image[::-1, :, :]
    if return_depth:
        depth = np.zeros((height, width), dtype='f4')
        depth_tex.read_into(depth)
        depth = depth[::-1, :]
    else:
        depth = None

    # Release
    vao.release()
    ibo.release()
    vbo_vertices.release()
    vbo_attr.release()
    fbo.release()
    image_tex.release()
    depth_tex.release()

    return image, depth


def texture(
    ctx: RastContext,
    uv: np.ndarray,
    texture: np.ndarray,
    interpolation: str= 'linear', 
    wrap: str = 'clamp'
) -> np.ndarray:
    """
    Given an UV image, texturing from the texture map
    """
    assert len(texture.shape) == 3 and 1 <= texture.shape[2] <= 4
    assert uv.shape[2] == 2
    height, width = uv.shape[:2]
    texture_dtype = map_np_dtype(texture.dtype)

    # Create VAO
    screen_quad_vbo = ctx.mgl_ctx.buffer(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype='f4'))
    screen_quad_ibo = ctx.mgl_ctx.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
    screen_quad_vao = ctx.mgl_ctx.vertex_array(ctx.program_texture(texture.shape[2]), [(screen_quad_vbo, '2f4', 'in_vert')], index_buffer=screen_quad_ibo, index_element_size=4)

    # Create texture, set filter and bind. TODO: min mag filter, mipmap
    texture_tex = ctx.mgl_ctx.texture((texture.shape[1], texture.shape[0]), texture.shape[2], dtype=texture_dtype, data=np.ascontiguousarray(texture))
    if interpolation == 'linear':
        texture_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    elif interpolation == 'nearest':
        texture_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    texture_tex.use(location=0)
    texture_uv = ctx.mgl_ctx.texture((width, height), 2, dtype='f4', data=np.ascontiguousarray(uv.astype('f4', copy=False)))
    texture_uv.filter = (moderngl.NEAREST, moderngl.NEAREST)
    texture_uv.use(location=1)

    # Create render buffer and frame buffer
    rb = ctx.mgl_ctx.renderbuffer((uv.shape[1], uv.shape[0]), texture.shape[2], dtype=texture_dtype)
    fbo = ctx.mgl_ctx.framebuffer(color_attachments=[rb])

    # Render
    fbo.use()
    fbo.viewport = (0, 0, width, height)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.BLEND)
    screen_quad_vao.render()

    # Read buffer
    image_buffer = np.frombuffer(fbo.read(components=texture.shape[2], attachment=0, dtype=texture_dtype), dtype=texture_dtype).reshape((height, width, texture.shape[2]))

    # Release
    texture_tex.release()
    rb.release()
    fbo.release()

    return image_buffer


def warp_image_by_depth(
    ctx: RastContext,
    src_depth: np.ndarray,
    src_image: np.ndarray = None,
    width: int = None,
    height: int = None,
    *,
    extrinsics_src: np.ndarray = None,
    extrinsics_tgt: np.ndarray = None,
    intrinsics_src: np.ndarray = None,
    intrinsics_tgt: np.ndarray = None,
    near: float = 0.1,
    far: float = 100.0,
    cull_backface: bool = True,
    ssaa: int = 1,
    return_depth: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Warp image by depth map.

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
        tgt_depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    assert src_depth.ndim == 2

    if width is None:
        width = src_depth.shape[1]
    if height is None:
        height = src_depth.shape[0]
    if src_image is not None:
        assert src_image.shape[-2:] == src_depth.shape[-2:], f'Shape of source image {src_image.shape} does not match shape of source depth {src_depth.shape}'

    # set up default camera parameters
    extrinsics_src = np.eye(4) if extrinsics_src is None else extrinsics_src
    extrinsics_tgt = np.eye(4) if extrinsics_tgt is None else extrinsics_tgt
    intrinsics_src = intrinsics_tgt if intrinsics_src is None else intrinsics_src
    intrinsics_tgt = intrinsics_src if intrinsics_tgt is None else intrinsics_tgt
    
    assert all(x is not None for x in [extrinsics_src, extrinsics_tgt, intrinsics_src, intrinsics_tgt]), "Make sure you have provided all the necessary camera parameters."

    # check shapes
    assert extrinsics_src.shape == (4, 4) and extrinsics_tgt.shape == (4, 4)
    assert intrinsics_src.shape == (3, 3) and intrinsics_tgt.shape == (3, 3) 

    # convert to view and perspective matrices
    view_tgt = transforms.extrinsics_to_view(extrinsics_tgt)
    perspective_tgt = transforms.intrinsics_to_perspective(intrinsics_tgt, near=near, far=far)

    # unproject depth map
    uv, faces = utils.image_mesh(*src_depth.shape[-2:])
    pts = transforms.unproject_cv(uv, src_depth.reshape(-1), extrinsics_src, intrinsics_src)
    faces = mesh.triangulate(faces, vertices=pts)

    # rasterize attributes
    if src_image is not None:
        attr = src_image.reshape(-1, src_image.shape[-1])
    else:
        attr = uv

    tgt_image, tgt_depth = rasterize_triangle_faces(
        ctx,
        pts,
        faces,
        attr,
        width * ssaa,
        height * ssaa,
        transform=perspective_tgt @ view_tgt,
        cull_backface=cull_backface,
        return_depth=return_depth,
    )

    if ssaa > 1:
        tgt_image = tgt_image.reshape(height, ssaa, width, ssaa, -1).mean(axis=(1, 3))
        tgt_depth = tgt_depth.reshape(height, ssaa, width, ssaa, -1).mean(axis=(1, 3)) if return_depth else None

    return tgt_image, tgt_depth

def test_rasterization(ctx: RastContext):
    """
    Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file.
    """
    vertices, faces = utils.cube(tri=True)
    attr = np.random.rand(len(vertices), 3).astype(np.float32)
    perspective = transforms.perspective(np.deg2rad(60), 1, 0.01, 100)
    view = transforms.view_look_at(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    image, depth = rasterize_triangle_faces(
        ctx, 
        vertices, 
        faces, 
        attr, 
        512, 512, 
        transform=(perspective @ view).astype(np.float32), 
        cull_backface=False,
        return_depth=True,
    )   
    import cv2
    cv2.imwrite('CHECKME.png', cv2.cvtColor((image.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    