import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from pathlib import Path
from typing import *
import itertools
import warnings

import click


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Inference script for wrapped baselines methods')
@click.option('--baseline', 'baseline_code_path', required=True, type=click.Path(), help='Path to the baseline model python code.')
@click.option('--input', '-i', 'input_path', type=str, required=True, help='Input image or folder')
@click.option('--output', '-o', 'output_path', type=str, default='./output', help='Output folder')
@click.option('--size', 'image_size', type=int, default=None, help='Resize input image')
@click.option('--skip', is_flag=True, help='Skip existing output')
@click.option('--maps', 'save_maps_', is_flag=True, help='Save output point / depth maps')
@click.option('--ply', 'save_ply_', is_flag=True, help='Save mesh in PLY format')
@click.option('--glb', 'save_glb_', is_flag=True, help='Save mesh in GLB format')
@click.option('--threshold', type=float, default=0.03, help='Depth edge detection threshold for saving mesh')
@click.pass_context
def main(ctx: click.Context, baseline_code_path: str, input_path: str, output_path: str, image_size: int, skip: bool, save_maps_, save_ply_: bool, save_glb_: bool, threshold: float):
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import utils3d

    from moge.utils.io import save_ply, save_glb
    from moge.utils.geometry_numpy import intrinsics_to_fov_numpy
    from moge.utils.vis import colorize_depth, colorize_depth_affine, colorize_disparity
    from moge.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module
    from moge.test.baseline import MGEBaselineInterface

    # Load the baseline model
    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type[MGEBaselineInterface] = getattr(module, 'Baseline')
    baseline : MGEBaselineInterface = baseline_cls.load.main(ctx.args, standalone_mode=False)

    # Input images list
    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Defaults to saving maps only. Please use "--maps", "--glb", or "--ply" to specify the output.')
        save_maps_ = True

    for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
        # Load one image at a time  
        image_np = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image_np.shape[:2]
        if image_size is not None and max(image_np.shape[:2]) > image_size:
            height, width = min(image_size, int(image_size * height / width)), min(image_size, int(image_size * width / height))
            image_np = cv2.resize(image_np, (width, height), cv2.INTER_AREA)
        image = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(baseline.device)
    
        # Inference  
        torch.cuda.synchronize()
        with torch.inference_mode(), (timer := timeit('Inference', verbose=False, average=True)):
            output = baseline.infer(image)
            torch.cuda.synchronize()
        
        inference_time = timer.average_time
        pbar.set_postfix({'average inference time': f'{inference_time:.3f}s'})

        # Save the output
        save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        if skip and save_path.exists():
            continue
        save_path.mkdir(parents=True, exist_ok=True)

        if save_maps_:
            cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if 'mask' in output:
                mask = output['mask'].cpu().numpy()
                cv2.imwrite(str(save_path /'mask.png'), (mask * 255).astype(np.uint8))  

            for k in ['points_metric', 'points_scale_invariant', 'points_affine_invariant']:
                if k in output:
                    points = output[k].cpu().numpy()
                    cv2.imwrite(str(save_path / f'{k}.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

            for k in ['depth_metric', 'depth_scale_invariant', 'depth_affine_invariant', 'disparity_affine_invariant']:
                if k in output:
                    depth = output[k].cpu().numpy()
                    cv2.imwrite(str(save_path / f'{k}.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    if k in ['depth_metric', 'depth_scale_invariant']:
                        depth_vis = colorize_depth(depth)
                    elif k == 'depth_affine_invariant':
                        depth_vis = colorize_depth_affine(depth)
                    elif k == 'disparity_affine_invariant':
                        depth_vis = colorize_disparity(depth)
                    cv2.imwrite(str(save_path / f'{k}_vis.png'), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
                
            if 'intrinsics' in output:
                intrinsics = output['intrinsics'].cpu().numpy()
                fov_x, fov_y = intrinsics_to_fov_numpy(intrinsics)
                with open(save_path / 'fov.json', 'w') as f:
                    json.dump({
                        'fov_x': float(np.rad2deg(fov_x)), 
                        'fov_y': float(np.rad2deg(fov_y)), 
                        'intrinsics': intrinsics.tolist()
                    }, f, indent=4)
        
        # Export mesh & visulization
        if save_ply_ or save_glb_:
            assert any(k in output for k in ['points_metric', 'points_scale_invariant', 'points_affine_invariant']), 'No point map found in output'
            points = next(output[k] for k in ['points_metric', 'points_scale_invariant', 'points_affine_invariant'] if k in output).cpu().numpy()
            mask = output['mask'] if 'mask' in output else np.ones_like(points[..., 0], dtype=bool)
            normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                points,
                image_np.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=width, height=height),
                mask=mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                tri=True
            )
            # When exporting the model, follow the OpenGL coordinate conventions:
            # - world coordinate system: x right, y up, z backward.
            # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
        
        if save_glb_:
            save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image_np)

        if save_ply_:
            save_ply(save_path / 'mesh.ply', vertices, faces, vertex_colors)

if __name__ == '__main__':
    main()
