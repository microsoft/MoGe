import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import time
import uuid
import tempfile
from typing import Union
import atexit
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import cv2
import torch
import numpy as np
import click
import trimesh
import trimesh.visual
from PIL import Image

from moge.model import MoGeModel
from moge.utils.vis import colorize_depth
import utils3d

model = MoGeModel.from_pretrained('Ruicheng/moge-vitl').cuda().eval()
thread_pool_executor = ThreadPoolExecutor(max_workers=1)


def delete_later(path: Union[str, os.PathLike], delay: int = 300):
    def _delete():
        try: 
            os.remove(path) 
        except: 
            pass
    def _wait_and_delete():
        time.sleep(delay)
        _delete(path)
    thread_pool_executor.submit(_wait_and_delete)
    atexit.register(_delete)


def run(image: np.ndarray, remove_edge: bool = True, max_size: int = 800):
    run_id = str(uuid.uuid4())

    larger_size = max(image.shape[:2])
    if larger_size > max_size:
        scale = max_size / larger_size
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width = image.shape[:2]

    image_tensor = torch.tensor(image, dtype=torch.float32, device=torch.device('cuda')).permute(2, 0, 1) / 255
    output = model.infer(image_tensor, resolution_level=9, apply_mask=True)
    points, depth, mask = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy()
    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)

    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        points,
        image.astype(np.float32) / 255,
        utils3d.numpy.image_uv(width=width, height=height),
        mask=mask & ~(utils3d.numpy.depth_edge(depth, rtol=0.03, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

    tempdir = Path(tempfile.gettempdir(), 'moge')
    tempdir.mkdir(exist_ok=True)

    output_glb_path = Path(tempdir, f'{run_id}.glb')
    output_glb_path.parent.mkdir(exist_ok=True)
    trimesh.Trimesh(
        vertices=vertices * [-1, 1, -1],    # No idea why Gradio 3D Viewer' default camera is flipped
        faces=faces, 
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs, 
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(image),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
    ).export(output_glb_path)

    output_ply_path = Path(tempdir, f'{run_id}.ply')
    output_ply_path.parent.mkdir(exist_ok=True)
    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(output_ply_path)

    colorized_depth = colorize_depth(depth)

    delete_later(output_glb_path, delay=300)
    delete_later(output_ply_path, delay=300)
        
    return colorized_depth, output_glb_path, output_ply_path.as_posix()


DESCRIPTION = """
## Turn a 2D image into a 3D point map with [MoGe](https://wangrc.site/MoGePage/)

NOTE: 
* The maximum size is set to 800px for efficiency purpose. Oversized images will be downsampled.
* The color in the 3D viewer may look dark due to rendering of 3D viewer. You may download the 3D model as .glb or .ply file to view it in other 3D viewers.
"""

@click.command()
@click.option('--share', is_flag=True, help='Whether to run the app in shared mode.')
def main(share: bool):
    gr.Interface(
        fn=run,
        inputs=[
            gr.Image(type="numpy", image_mode="RGB"),
            gr.Checkbox(True, label="Remove edges"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Depth map (colorized)"),
            gr.Model3D(display_mode="solid", clear_color=[1.0, 1.0, 1.0, 1.0], label="3D Viewer"),
            gr.File(type="filepath", label="Download the model as .ply file"),
        ],
        title=None,
        description=DESCRIPTION,
        clear_btn=None,
        allow_flagging="never",
        theme=gr.themes.Soft()
    ).launch(share=share)


if __name__ == '__main__':
    main()