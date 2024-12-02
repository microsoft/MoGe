from typing import *

import torch
import nvdiffrast.torch as dr

from . import utils, transforms, mesh
from ._helpers import batched


__all__ = [
    'RastContext',
    'rasterize_triangle_faces', 
    'warp_image_by_depth',
    'warp_image_by_forward_flow',
]


class RastContext:
    """
    Create a rasterization context. Nothing but a wrapper of nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext.
    """
    def __init__(self, nvd_ctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = None, *, backend: Literal['cuda', 'gl'] = 'gl',  device: Union[str, torch.device] = None):
        if nvd_ctx is not None:
            self.nvd_ctx = nvd_ctx
            return 
        
        if backend == 'gl':
            self.nvd_ctx = dr.RasterizeGLContext(device=device)
        elif backend == 'cuda':
            self.nvd_ctx = dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f'Unknown backend: {backend}')


def rasterize_triangle_faces(
    ctx: RastContext,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
    attr: torch.Tensor = None,
    uv: torch.Tensor = None,
    texture: torch.Tensor = None,
    model: torch.Tensor = None,
    view: torch.Tensor = None,
    projection: torch.Tensor = None,
    antialiasing: Union[bool, List[int]] = True,
    diff_attrs: Union[None, List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Rasterize a mesh with vertex attributes.

    Args:
        ctx (GLContext): rasterizer context
        vertices (np.ndarray): (B, N, 2 or 3 or 4)
        faces (torch.Tensor): (T, 3)
        width (int): width of the output image
        height (int): height of the output image
        attr (torch.Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
        uv (torch.Tensor, optional): (B, N, 2) uv coordinates. Defaults to None.
        texture (torch.Tensor, optional): (B, H, W, C) texture. Defaults to None.
        model (torch.Tensor, optional): ([B,] 4, 4) model matrix. Defaults to None (identity).
        view (torch.Tensor, optional): ([B,] 4, 4) view matrix. Defaults to None (identity).
        projection (torch.Tensor, optional): ([B,] 4, 4) projection matrix. Defaults to None (identity).
        antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
        diff_attrs (Union[None, List[int]], optional): indices of attributes to compute screen-space derivatives. Defaults to None.

    Returns:
        Dictionary containing:
          - image: (torch.Tensor): (B, C, H, W)
          - depth: (torch.Tensor): (B, H, W) screen space depth, ranging from 0 (near) to 1. (far)
                   NOTE: Empty pixels will have depth 1., i.e. far plane.
          - mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
          - image_dr: (torch.Tensor): (B, 4, H, W) screen space derivatives of the attributes
          - face_id: (torch.Tensor): (B, H, W) face ids
          - uv: (torch.Tensor): (B, N, 2) uv coordinates (if uv is not None)
          - uv_dr: (torch.Tensor): (B, N, 4) uv derivatives (if uv is not None)
          - texture: (torch.Tensor): (B, H, W, C) texture (if uv and texture are not None)
    """
    assert vertices.ndim == 3
    assert faces.ndim == 2

    if vertices.shape[-1] == 2:
        vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]), torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 3:
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 4:
        pass
    else:
        raise ValueError(f'Wrong shape of vertices: {vertices.shape}')
    
    mvp = projection if projection is not None else torch.eye(4).to(vertices)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    
    pos_clip = vertices @ mvp.transpose(-1, -2)
    faces = faces.contiguous()
    if attr is not None:
        attr = attr.contiguous()
    
    rast_out, rast_db = dr.rasterize(ctx.nvd_ctx, pos_clip, faces, resolution=[height, width], grad_db=True)
    face_id = rast_out[..., 3].flip(1)
    depth = rast_out[..., 2].flip(1)
    mask = (face_id > 0).float()
    depth = (depth * 0.5 + 0.5) * mask + (1.0 - mask)

    ret = {
        'depth': depth,
        'mask': mask,
        'face_id': face_id,
    }

    if attr is not None:
        image, image_dr = dr.interpolate(attr, rast_out, faces, rast_db, diff_attrs=diff_attrs)
        if antialiasing == True:
            image = dr.antialias(image, rast_out, pos_clip, faces)
        elif isinstance(antialiasing, list):
            aa_image = dr.antialias(image[..., antialiasing], rast_out, pos_clip, faces)
            image[..., antialiasing] = aa_image
        image = image.flip(1).permute(0, 3, 1, 2)
        ret['image'] = image

    if uv is not None:
        uv_map, uv_map_dr = dr.interpolate(uv, rast_out, faces, rast_db, diff_attrs='all')
        ret['uv'] = uv_map
        ret['uv_dr'] = uv_map_dr
        if texture is not None:
            texture_map = dr.texture(ctx.nvd_ctx, uv_map, uv_map_dr)
            ret['texture'] = texture_map.flip(1).permute(0, 3, 1, 2)

    if diff_attrs is not None:
        image_dr = image_dr.flip(1).permute(0, 3, 1, 2)
        ret['image_dr'] = image_dr

    return ret


def texture(
    ctx: RastContext,
    uv: torch.Tensor,
    uv_da: torch.Tensor,
    texture: torch.Tensor,
) -> torch.Tensor:
    dr.texture(ctx.nvd_ctx, uv, texture)


def warp_image_by_depth(
    ctx: RastContext,
    depth: torch.FloatTensor,
    image: torch.FloatTensor = None,
    mask: torch.BoolTensor = None,
    width: int = None,
    height: int = None,
    *,
    extrinsics_src: torch.FloatTensor = None,
    extrinsics_tgt: torch.FloatTensor = None,
    intrinsics_src: torch.FloatTensor = None,
    intrinsics_tgt: torch.FloatTensor = None,
    near: float = 0.1,
    far: float = 100.0,
    antialiasing: bool = True,
    backslash: bool = False,
    padding: int = 0,
    return_uv: bool = False,
    return_dr: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
    """
    Warp image by depth. 
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
        dr: (torch.FloatTensor): (B, 4, H, W) image-space derivatives of uv
    """
    assert depth.ndim == 3
    batch_size = depth.shape[0]

    if width is None:
        width = depth.shape[-1]
    if height is None:
        height = depth.shape[-2]
    if image is not None:
        assert image.shape[-2:] == depth.shape[-2:], f'Shape of image {image.shape} does not match shape of depth {depth.shape}'

    if extrinsics_src is None:
        extrinsics_src = torch.eye(4).to(depth)
    if extrinsics_tgt is None:
        extrinsics_tgt = torch.eye(4).to(depth)
    if intrinsics_src is None:
        intrinsics_src = intrinsics_tgt
    if intrinsics_tgt is None:
        intrinsics_tgt = intrinsics_src
    
    assert all(x is not None for x in [extrinsics_src, extrinsics_tgt, intrinsics_src, intrinsics_tgt]), "Make sure you have provided all the necessary camera parameters."

    view_tgt = transforms.extrinsics_to_view(extrinsics_tgt)
    perspective_tgt = transforms.intrinsics_to_perspective(intrinsics_tgt, near=near, far=far)

    if padding > 0:
        uv, faces = utils.image_mesh(width=width+2, height=height+2)
        uv = (uv - 1 / (width + 2)) * ((width + 2) / width)
        uv_ = uv.clone().reshape(height+2, width+2, 2)
        uv_[0, :, 1] -= padding / height
        uv_[-1, :, 1] += padding / height
        uv_[:, 0, 0] -= padding / width
        uv_[:, -1, 0] += padding / width
        uv_ = uv_.reshape(-1, 2)
        depth = torch.nn.functional.pad(depth, [1, 1, 1, 1], mode='replicate')
        if image is not None:
            image = torch.nn.functional.pad(image, [1, 1, 1, 1], mode='replicate')
        uv, uv_, faces = uv.to(depth.device), uv_.to(depth.device), faces.to(depth.device)
        pts = transforms.unproject_cv(
            uv_,
            depth.flatten(-2, -1),
            extrinsics_src,
            intrinsics_src,
        )
    else:    
        uv, faces = utils.image_mesh(width=depth.shape[-1], height=depth.shape[-2])
        if mask is not None:
            depth = torch.where(mask, depth, torch.tensor(far, dtype=depth.dtype, device=depth.device))
        uv, faces = uv.to(depth.device), faces.to(depth.device)
        pts = transforms.unproject_cv(
            uv,
            depth.flatten(-2, -1),
            extrinsics_src,
            intrinsics_src,
        )

    # triangulate
    if batch_size == 1:
        faces = mesh.triangulate(faces, vertices=pts[0])
    else:
        faces = mesh.triangulate(faces, backslash=backslash)

    # rasterize attributes
    diff_attrs = None
    if image is not None:
        attr = image.permute(0, 2, 3, 1).flatten(1, 2)
        if return_dr or return_uv:
            if return_dr:
                diff_attrs = [image.shape[1], image.shape[1]+1]
            if return_uv and antialiasing:
                antialiasing = list(range(image.shape[1]))
            attr = torch.cat([attr, uv.expand(batch_size, -1, -1)], dim=-1)
    else:
        attr = uv.expand(batch_size, -1, -1)
        if antialiasing:
            print("\033[93mWarning: you are performing antialiasing on uv. This may cause artifacts.\033[0m")
        if return_uv:
            return_uv = False
            print("\033[93mWarning: image is None, return_uv is ignored.\033[0m")
        if return_dr:
            diff_attrs = [0, 1]

    if mask is not None:
        attr = torch.cat([attr, mask.float().flatten(1, 2).unsqueeze(-1)], dim=-1)

    rast = rasterize_triangle_faces(
        ctx,
        pts,
        faces,
        width,
        height,
        attr=attr,
        view=view_tgt,
        perspective=perspective_tgt,
        antialiasing=antialiasing,
        diff_attrs=diff_attrs,
    )
    if return_dr:
        output_image, screen_depth, output_dr = rast['image'], rast['depth'], rast['image_dr']
    else:
        output_image, screen_depth = rast['image'], rast['depth']
    output_mask = screen_depth < 1.0

    if mask is not None:
        output_image, rast_mask = output_image[..., :-1, :, :], output_image[..., -1, :, :]
        output_mask &= (rast_mask > 0.9999).reshape(-1, height, width)

    if (return_dr or return_uv) and image is not None:
        output_image, output_uv = output_image[..., :-2, :, :], output_image[..., -2:, :, :]

    output_depth = transforms.depth_buffer_to_linear(screen_depth, near=near, far=far) * output_mask
    output_image = output_image * output_mask.unsqueeze(1)

    outs = [output_image, output_depth, output_mask]
    if return_uv:
        outs.append(output_uv)
    if return_dr:
        outs.append(output_dr)
    return tuple(outs)


def warp_image_by_forward_flow(
    ctx: RastContext,
    image: torch.FloatTensor,
    flow: torch.FloatTensor,
    depth: torch.FloatTensor = None,
    *,
    antialiasing: bool = True,
    backslash: bool = False,
) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
    """
    Warp image by forward flow.
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
        mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
    """
    assert image.ndim == 4, f'Wrong shape of image: {image.shape}'
    batch_size, _, height, width = image.shape

    if depth is None:
        depth = torch.ones_like(flow[:, 0])

    extrinsics = torch.eye(4).to(image)
    fov = torch.deg2rad(torch.tensor([45.0], device=image.device))
    intrinsics = transforms.intrinsics_from_fov(fov, width, height, normalize=True)[0] 
   
    view = transforms.extrinsics_to_view(extrinsics)
    perspective = transforms.intrinsics_to_perspective(intrinsics, near=0.1, far=100)

    uv, faces = utils.image_mesh(width=width, height=height)
    uv, faces = uv.to(image.device), faces.to(image.device)
    uv = uv + flow.permute(0, 2, 3, 1).flatten(1, 2)
    pts = transforms.unproject_cv(
        uv,
        depth.flatten(-2, -1),
        extrinsics,
        intrinsics,
    )

    # triangulate
    if batch_size == 1:
        faces = mesh.triangulate(faces, vertices=pts[0])
    else:
        faces = mesh.triangulate(faces, backslash=backslash)

    # rasterize attributes
    attr = image.permute(0, 2, 3, 1).flatten(1, 2)
    rast = rasterize_triangle_faces(
        ctx,
        pts,
        faces,
        width,
        height,
        attr=attr,
        view=view,
        perspective=perspective,
        antialiasing=antialiasing,
    )
    output_image, screen_depth = rast['image'], rast['depth']
    output_mask = screen_depth < 1.0
    output_image = output_image * output_mask.unsqueeze(1)

    outs = [output_image, output_mask]
    return tuple(outs)
