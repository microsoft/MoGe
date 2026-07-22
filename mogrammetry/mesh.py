"""
Mesh generation and texturing from point clouds.
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image


class MeshGenerator:
    """Generate textured meshes from point clouds."""

    def __init__(
        self,
        method: str = 'poisson',
        poisson_depth: int = 9,
        poisson_width: float = 0.0,
        poisson_scale: float = 1.1,
        poisson_linear_fit: bool = False,
        ball_pivoting_radii: Optional[List[float]] = None,
        alpha: float = 0.03,
        simplify_mesh: bool = False,
        target_faces: Optional[int] = None
    ):
        """
        Initialize mesh generator.

        Args:
            method: Meshing method ('poisson', 'ball_pivoting', 'alpha_shape')
            poisson_depth: Depth for Poisson reconstruction
            poisson_width: Width for Poisson reconstruction
            poisson_scale: Scale for Poisson reconstruction
            poisson_linear_fit: Use linear fit for Poisson
            ball_pivoting_radii: Radii for ball pivoting algorithm
            alpha: Alpha value for alpha shape
            simplify_mesh: Whether to simplify mesh
            target_faces: Target number of faces for simplification
        """
        self.method = method
        self.poisson_depth = poisson_depth
        self.poisson_width = poisson_width
        self.poisson_scale = poisson_scale
        self.poisson_linear_fit = poisson_linear_fit
        self.ball_pivoting_radii = ball_pivoting_radii or [0.005, 0.01, 0.02, 0.04]
        self.alpha = alpha
        self.simplify_mesh = simplify_mesh
        self.target_faces = target_faces

    def generate_mesh(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """
        Generate mesh from point cloud.

        Args:
            pcd: Input point cloud

        Returns:
            mesh: Generated mesh
            stats: Statistics about mesh generation
        """
        stats = {
            'method': self.method,
            'input_points': len(pcd.points)
        }

        # Ensure normals are estimated
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        # Generate mesh based on method
        if self.method == 'poisson':
            mesh, densities = self._poisson_reconstruction(pcd)
            stats['densities'] = densities
        elif self.method == 'ball_pivoting':
            mesh = self._ball_pivoting(pcd)
        elif self.method == 'alpha_shape':
            mesh = self._alpha_shape(pcd)
        else:
            raise ValueError(f"Unknown meshing method: {self.method}")

        if mesh is None:
            raise RuntimeError(f"Mesh generation failed with method: {self.method}")

        stats['vertices_before_cleanup'] = len(mesh.vertices)
        stats['faces_before_cleanup'] = len(mesh.triangles)

        # Clean up mesh
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()

        stats['vertices_after_cleanup'] = len(mesh.vertices)
        stats['faces_after_cleanup'] = len(mesh.triangles)

        # Simplify if requested
        if self.simplify_mesh and self.target_faces is not None:
            mesh = self._simplify_mesh(mesh, self.target_faces)
            stats['vertices_after_simplification'] = len(mesh.vertices)
            stats['faces_after_simplification'] = len(mesh.triangles)

        # Compute vertex normals
        mesh.compute_vertex_normals()

        return mesh, stats

    def _poisson_reconstruction(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """Poisson surface reconstruction."""
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.poisson_depth,
            width=self.poisson_width,
            scale=self.poisson_scale,
            linear_fit=self.poisson_linear_fit
        )

        # Remove low-density vertices (often outliers)
        if len(densities) > 0:
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh, densities

    def _ball_pivoting(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Ball pivoting algorithm."""
        radii = o3d.utility.DoubleVector(self.ball_pivoting_radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            radii
        )
        return mesh

    def _alpha_shape(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Alpha shape algorithm."""
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            self.alpha
        )
        return mesh

    def _simplify_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_faces: int
    ) -> o3d.geometry.TriangleMesh:
        """Simplify mesh to target face count."""
        current_faces = len(mesh.triangles)
        if current_faces <= target_faces:
            return mesh

        # Use quadric decimation
        mesh_simplified = mesh.simplify_quadric_decimation(target_faces)

        return mesh_simplified


class TextureMapper:
    """Map textures onto meshes from source images."""

    def __init__(
        self,
        texture_size: int = 4096,
        method: str = 'mvs'
    ):
        """
        Initialize texture mapper.

        Args:
            texture_size: Size of texture atlas
            method: Texturing method ('average', 'max_weight', 'mvs')
        """
        self.texture_size = texture_size
        self.method = method

    def create_textured_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        images: List[np.ndarray],
        camera_matrices: List[np.ndarray],
        camera_intrinsics: List[np.ndarray]
    ) -> trimesh.Trimesh:
        """
        Create textured mesh by projecting images onto mesh.

        Args:
            mesh: Input mesh
            images: List of source images (RGB, 0-1)
            camera_matrices: List of camera-to-world matrices
            camera_intrinsics: List of camera intrinsic matrices

        Returns:
            Textured trimesh
        """
        # Convert Open3D mesh to trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        # Create UV coordinates (simple planar projection for now)
        # In production, would use proper UV unwrapping
        uvs = self._generate_uv_coordinates(vertices)

        # Project images onto mesh
        texture = self._project_images_to_texture(
            vertices,
            faces,
            uvs,
            images,
            camera_matrices,
            camera_intrinsics
        )

        # Create trimesh with texture
        material = trimesh.visual.material.SimpleMaterial(
            image=Image.fromarray((texture * 255).astype(np.uint8))
        )

        visual = trimesh.visual.TextureVisuals(
            uv=uvs,
            image=Image.fromarray((texture * 255).astype(np.uint8)),
            material=material
        )

        textured_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=visual,
            process=False
        )

        return textured_mesh

    def _generate_uv_coordinates(self, vertices: np.ndarray) -> np.ndarray:
        """Generate UV coordinates for vertices."""
        # Simple box projection
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        extent = bbox_max - bbox_min

        # Normalize to [0, 1]
        if np.max(extent) > 0:
            uvs = (vertices - bbox_min) / np.max(extent)
            uvs = uvs[:, :2]  # Use X, Y for UV
        else:
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)

        return uvs

    def _project_images_to_texture(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: np.ndarray,
        images: List[np.ndarray],
        camera_matrices: List[np.ndarray],
        camera_intrinsics: List[np.ndarray]
    ) -> np.ndarray:
        """Project images onto texture atlas."""
        # Create blank texture
        texture = np.zeros((self.texture_size, self.texture_size, 3), dtype=np.float32)
        weight_map = np.zeros((self.texture_size, self.texture_size), dtype=np.float32)

        # For each vertex, find best image view
        for img, cam_matrix, intrinsic in zip(images, camera_matrices, camera_intrinsics):
            # Project vertices to image
            vertices_homo = np.hstack([vertices, np.ones((len(vertices), 1))])

            # World to camera
            cam_inv = np.linalg.inv(cam_matrix)
            vertices_cam = (cam_inv @ vertices_homo.T).T
            vertices_cam = vertices_cam[:, :3]

            # Camera to image
            vertices_img = (intrinsic @ vertices_cam.T).T
            vertices_img = vertices_img[:, :2] / (vertices_cam[:, 2:3] + 1e-6)

            # Check if vertices are visible
            H, W = img.shape[:2]
            visible = (
                (vertices_cam[:, 2] > 0) &
                (vertices_img[:, 0] >= 0) & (vertices_img[:, 0] < W) &
                (vertices_img[:, 1] >= 0) & (vertices_img[:, 1] < H)
            )

            # Sample colors from image
            for i, (uv, img_coord) in enumerate(zip(uvs, vertices_img)):
                if not visible[i]:
                    continue

                # UV to texture coordinates
                tex_u = int(uv[0] * (self.texture_size - 1))
                tex_v = int(uv[1] * (self.texture_size - 1))

                # Image coordinates
                img_u = int(img_coord[0])
                img_v = int(img_coord[1])

                if 0 <= img_u < W and 0 <= img_v < H:
                    color = img[img_v, img_u]
                    weight = 1.0  # Could compute based on viewing angle

                    # Blend with existing texture
                    texture[tex_v, tex_u] += color * weight
                    weight_map[tex_v, tex_u] += weight

        # Normalize by weights
        valid_mask = weight_map > 0
        texture[valid_mask] /= weight_map[valid_mask, None]

        return texture


def save_mesh(
    mesh: o3d.geometry.TriangleMesh,
    output_path: str,
    format: str = 'ply'
):
    """
    Save mesh to file.

    Args:
        mesh: Mesh to save
        output_path: Output file path
        format: Output format ('ply', 'obj', 'stl', 'glb')
    """
    if format in ['ply', 'obj', 'stl']:
        o3d.io.write_triangle_mesh(output_path, mesh)
    elif format == 'glb':
        # Convert to trimesh for GLB export
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
            tm = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False
            )
        else:
            tm = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False
            )

        tm.export(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def mesh_to_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 100000
) -> o3d.geometry.PointCloud:
    """Sample point cloud from mesh."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd
