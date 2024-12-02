import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parents[1]))

from typing import *
import itertools
import json
import warnings

import cv2
import numpy as np
from numpy import ndarray
import torch
from PIL import Image
from tqdm import tqdm, trange
import trimesh
import trimesh.visual
import click
from scipy.sparse import csr_array, hstack, vstack
from scipy.ndimage import convolve
from scipy.sparse.linalg import lsmr

from moge.model import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth
import utils3d


def get_panorama_cameras():
    vertices, _ = utils3d.numpy.icosahedron()
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, resolution: int):
    height, width = image.shape[:2]
    uv = utils3d.numpy.image_uv(width=resolution, height=resolution)
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)

        splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)    
        splitted_images.append(splitted_image)
    return splitted_images


def poisson_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, ndarray]:
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode='wrap' if wrap_x else 'edge')
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode='wrap' if wrap_y else 'edge')
    
    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack([
        grid_index[1:-1, 1:-1],
        grid_index[:-2, 1:-1],         # up
        grid_index[2:, 1:-1],          # down
        grid_index[1:-1, :-2],         # left
        grid_index[1:-1, 2:]           # right
    ], axis=-1).reshape(-1)                                                                 
    indptr = np.arange(0, height * width * 5 + 1, 5) 
    A = csr_array((data, indices, indptr), shape=(height * width, height * width))
    
    return A


def grad_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, np.ndarray]:
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')

    data = np.concatenate([
        np.concatenate([
            np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),       # x[i,j-1]           
        ], axis=1).reshape(-1),
        np.concatenate([
            np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),       # x[i-1,j]           
        ], axis=1).reshape(-1),
    ])
    indices = np.concatenate([
        np.concatenate([
            grid_index[:, :-1].reshape(-1, 1),
            grid_index[:, 1:].reshape(-1, 1),
        ], axis=1).reshape(-1),
        np.concatenate([
            grid_index[:-1, :].reshape(-1, 1),
            grid_index[1:, :].reshape(-1, 1),
        ], axis=1).reshape(-1),
    ])
    indptr = np.arange(0, grid_index.shape[0] * (grid_index.shape[1] - 1) * 2 + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2 + 1, 2)
    A = csr_array((data, indices, indptr), shape=(grid_index.shape[0] * (grid_index.shape[1] - 1) + (grid_index.shape[0] - 1) * grid_index.shape[1], height * width))

    return A


def merge_panorama_depth(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray]):
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics)
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    uv = utils3d.numpy.image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.numpy.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        
        projected_pixels = utils3d.numpy.uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)
        
        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(projection_valid_mask, cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_pred_mask = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]
        
        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)
        
        panorama_pred_masks.append(panorama_pred_mask)  
        
    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    # Solve overdetermined system
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask], 
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask]
    ])
    x, *_ = lsmr(
        A, b, 
        atol=1e-5, btol=1e-5,
        x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None, 
        show=False,
    )
    
    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask
         

@click.command(help='Inference script for the MoGe model.')
@click.option('--input', 'input_path', type=click.Path(exists=True), help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--output', 'output_path', type=click.Path(), help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl', help='Pretrained model name or path. Default is "Ruicheng/moge-vitl"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Default is "cuda"')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Default is None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level of inference. The higher, the better but slower. Default is 9. Note that it is irrelevant to the output resolution.')
@click.option('--threshold', type=float, default=0.03, help='Threshold for removing edges. Default is 0.03. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--batch_size', type=int, default=4, help='Batch size for inference. Default is 4.')
@click.option('--splitted', 'save_splitted', is_flag=True, help='Whether to save the splitted images. Default is False.')
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps and fov(image, depth, mask, points, fov).')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a.glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a.ply file. The color will be saved as vertex colors.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')
def main(
    input_path: str,
    output_path: str,
    pretrained_model_name_or_path: str,
    device_name: str,
    resize_to: int,
    resolution_level: int,
    threshold: float,
    batch_size: int,
    save_splitted: bool,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):  
    device = torch.device(device_name)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Please use "--maps", "--glb", or "--ply" to specify the output.')

    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    for image_path in (pbar := tqdm(image_paths, desc='Total images', disable=len(image_paths) <= 1)):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        
        splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
        splitted_resolution = 512
        splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution)

        # Infer each view 
        print('Inferring...') if pbar.disable else pbar.set_postfix_str(f'Inferring')

        splitted_distance_maps, splitted_masks = [], []
        for i in trange(0, len(splitted_images), batch_size, desc='Inferring splitted views', disable=len(splitted_images) <= batch_size, leave=False):
            image_tensor = torch.tensor(np.stack(splitted_images[i:i + batch_size]) / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            fov_x, fov_y = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(splitted_intriniscs[i:i + batch_size])))
            fov_x = torch.tensor(fov_x, dtype=torch.float32, device=device)
            output = model.infer(image_tensor, fov_x=fov_x, apply_mask=False)
            distance_map, mask = output['points'].norm(dim=-1).cpu().numpy(), output['mask'].cpu().numpy()
            splitted_distance_maps.extend(list(distance_map))
            splitted_masks.extend(list(mask))

        # Save splitted
        if save_splitted:
            splitted_save_path = Path(output_path, image_path.stem, 'splitted')
            splitted_save_path.mkdir(exist_ok=True, parents=True)
            for i in range(len(splitted_images)):
                cv2.imwrite(str(splitted_save_path / f'{i:02d}.jpg'), cv2.cvtColor(splitted_images[i], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(splitted_save_path / f'{i:02d}_distance_vis.png'), cv2.cvtColor(colorize_depth(splitted_distance_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))

        # Merge
        print('Merging...') if pbar.disable else pbar.set_postfix_str(f'Merging')

        merging_width, merging_height = min(1920, width), min(960, height)
        panorama_depth, panorama_mask = merge_panorama_depth(merging_width, merging_height, splitted_distance_maps, splitted_masks, splitted_extrinsics, splitted_intriniscs)
        panorama_depth = panorama_depth.astype(np.float32)
        panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
        panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0
        points = panorama_depth[:, :, None] * spherical_uv_to_directions(utils3d.numpy.image_uv(width=width, height=height))
        
        # Write outputs
        print('Writing outputs...') if pbar.disable else pbar.set_postfix_str(f'Inferring')
        save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        save_path.mkdir(exist_ok=True, parents=True)
        if save_maps_:
            cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(panorama_depth, mask=panorama_mask), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth.exr'), panorama_depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path / 'points.exr'), points, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path /'mask.png'), (panorama_mask * 255).astype(np.uint8))

        # Export mesh & visulization
        if save_glb_ or save_ply_ or show:
            normals, normals_mask = utils3d.numpy.points_to_normals(points, panorama_mask)
            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                points,
                image.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=width, height=height),
                mask=panorama_mask & ~(utils3d.numpy.depth_edge(panorama_depth, rtol=threshold) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                tri=True
            )

        if save_glb_:
            save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image)

        if save_ply_:
            save_ply(save_path / 'mesh.ply', vertices, faces, vertex_colors)

        if show:
            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                faces=faces, 
                process=False
            ).show()  


if __name__ == '__main__':
    main()