import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import json
import warnings

import click

         
@click.command(help='Inference script for panorama images')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--output', '-o', 'output_path', type=click.Path(), default='./output', help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl', help='Pretrained model name or path. Defaults to "Ruicheng/moge-vitl"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level of inference. The higher, the better but slower. Defaults to 9. Note that it is irrelevant to the output resolution.')
@click.option('--threshold', type=float, default=0.03, help='Threshold for removing edges. Defaults to 0.03. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--batch_size', type=int, default=4, help='Batch size for inference. Defaults to 4.')
@click.option('--splitted', 'save_splitted', is_flag=True, help='Whether to save the splitted images. Defaults to False.')
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
    # Lazy import
    import cv2
    import numpy as np
    from numpy import ndarray
    import torch
    from PIL import Image
    from tqdm import tqdm, trange
    import trimesh
    import trimesh.visual
    from scipy.sparse import csr_array, hstack, vstack
    from scipy.ndimage import convolve
    from scipy.sparse.linalg import lsmr

    import utils3d
    from moge.model.v1 import MoGeModel
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth
    from moge.utils.panorama import spherical_uv_to_directions, get_panorama_cameras, split_panorama_image, merge_panorama_depth

    
    device = torch.device(device_name)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    # Write outputs
    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", or "--ply" to specify the output.')
        save_maps_ = save_glb_ = save_ply_ = True

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