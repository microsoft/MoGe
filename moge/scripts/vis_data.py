import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import click


@click.command()
@click.argument('folder_or_path', type=click.Path(exists=True))
@click.option('--output', '-o', 'output_folder', type=click.Path(), help='Path to output folder')
@click.option('--max_depth', '-m', type=float, default=float('inf'), help='max depth')
@click.option('--fov', type=float, default=None, help='field of view in degrees')
@click.option('--show', 'show', is_flag=True, help='show point cloud')
@click.option('--depth', 'depth_filename', type=str, default='depth.png', help='depth image file name')
@click.option('--ply', 'save_ply', is_flag=True, help='save point cloud as PLY file')
@click.option('--depth_vis', 'save_depth_vis', is_flag=True, help='save depth image')
@click.option('--inf', 'inf_mask', is_flag=True, help='use infinity mask')
@click.option('--version', 'version', type=str, default='v3', help='version of rgbd data')
def main(
    folder_or_path: str,
    output_folder: str,
    max_depth: float,
    fov: float,
    depth_filename: str,
    show: bool,
    save_ply: bool,
    save_depth_vis: bool,
    inf_mask: bool,
    version: str
):  
    # Lazy import
    import cv2
    import numpy as np
    import utils3d
    from tqdm import tqdm
    import trimesh

    from moge.utils.io import read_image, read_depth, read_meta
    from moge.utils.vis import colorize_depth, colorize_normal

    filepaths = sorted(p.parent for p in Path(folder_or_path).rglob('meta.json')) 

    for filepath in tqdm(filepaths):
        image = read_image(Path(filepath, 'image.jpg'))
        depth, unit = read_depth(Path(filepath, depth_filename))
        meta = read_meta(Path(filepath,'meta.json'))
        depth_mask = np.isfinite(depth)
        depth_mask_inf = (depth == np.inf)
        intrinsics = np.array(meta['intrinsics'])

        extrinsics = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)   # OpenGL's identity camera
        verts = utils3d.numpy.unproject_cv(utils3d.numpy.image_uv(*image.shape[:2]), depth, extrinsics=extrinsics, intrinsics=intrinsics)
        
        depth_mask_ply = depth_mask & (depth < depth[depth_mask].min() * max_depth)
        point_cloud = trimesh.PointCloud(verts[depth_mask_ply], image[depth_mask_ply] / 255)
        
        if show:
            point_cloud.show()

        if output_folder is None:
            output_path = filepath
        else:
            output_path = Path(output_folder, filepath.name)
            output_path.mkdir(exist_ok=True, parents=True)

        if inf_mask:
            depth = np.where(depth_mask_inf, np.inf, depth)
            depth_mask = depth_mask | depth_mask_inf

        if save_depth_vis:
            p = output_path.joinpath('depth_vis.png')
            cv2.imwrite(str(p), cv2.cvtColor(colorize_depth(depth, depth_mask), cv2.COLOR_RGB2BGR))
            print(f"{p}")

        if save_ply:
            p = output_path.joinpath('pointcloud.ply')
            point_cloud.export(p)
            print(f"{p}")

if __name__ == '__main__':
    main()