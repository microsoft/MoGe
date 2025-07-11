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


@click.command(help='Inference script')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--fov_x', 'fov_x_', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
@click.option('--output', '-o', 'output_path', default='./output', type=click.Path(), help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=None, help='Pretrained model name or path. If not provided, the corresponding default model will be chosen.')
@click.option('--version', 'model_version', type=click.Choice(['v1', 'v2']), default='v2', help='Model version. Defaults to "v2"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for much faster inference.')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. \
Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
@click.option('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
`resolution_level` will be ignored if `num_tokens` is provided. Default: None')
@click.option('--threshold', type=float, default=0.04, help='Threshold for removing edges. Defaults to 0.01. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps (image, point map, depth map, normal map, mask) and fov.')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a.glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a.ply file. The color will be saved as vertex colors.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')
def main(
    input_path: str,
    fov_x_: float,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    resolution_level: int,
    num_tokens: int,
    threshold: float,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):  
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    import trimesh
    import trimesh.visual
    import click

    from moge.model import import_model_class_by_version
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
    import utils3d

    device = torch.device(device_name)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    if use_fp16:
        model.half()
    
    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", or "--ply" to specify the output.')
        save_maps_ = save_glb_ = save_ply_ = True

    for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        # Inference
        output = model.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        normal = output['normal'].cpu().numpy() if 'normal' in output else None

        save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        save_path.mkdir(exist_ok=True, parents=True)

        # Save images / maps
        if save_maps_:
            cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path / 'mask.png'), (mask * 255).astype(np.uint8))
            cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            if normal is not None:
                cv2.imwrite(str(save_path / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
            fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
            with open(save_path / 'fov.json', 'w') as f:
                json.dump({
                    'fov_x': round(float(np.rad2deg(fov_x)), 2),
                    'fov_y': round(float(np.rad2deg(fov_y)), 2),
                }, f)

        # Export mesh & visulization
        if save_glb_ or save_ply_ or show:
            mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold)
            if normal is None:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=mask_cleaned,
                    tri=True
                )
                vertex_normals = None
            else:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    normal,
                    mask=mask_cleaned,
                    tri=True
                )
            # When exporting the model, follow the OpenGL coordinate conventions:
            # - world coordinate system: x right, y up, z backward.
            # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
            if normal is not None:
                vertex_normals = vertex_normals * [1, -1, -1]

        if save_glb_:
            save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)

        if save_ply_:
            save_ply(save_path / 'pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

        if show:
            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                vertex_normals=vertex_normals,
                faces=faces, 
                process=False
            ).show()  


if __name__ == '__main__':
    main()
