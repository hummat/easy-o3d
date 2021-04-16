from enum import Flag, auto
from typing import Any, Union

import numpy as np
import open3d as o3d

PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic
Image = o3d.geometry.Image
PointCloud = o3d.geometry.PointCloud
TriangleMesh = o3d.geometry.TriangleMesh


class SampleStyle(Flag):
    UNIFORMLY = auto()
    POISSON_DISK = auto()


def eval_data(data: Any, **kwargs: Any) -> Any:
    """Convenience function automatically determines the data type and loads the data accordingly.

    Args:
        data (Any): The data to be loaded.
        kwargs (any): Optional additional keyword arguments.

    Returns:
        Any: The loaded data.
    """
    if isinstance(data, (Image, np.ndarray)) and "camera_intrinsics" in kwargs:
        return get_point_cloud_from_depth_image(image=data, **kwargs)
    elif isinstance(data, (str, TriangleMesh)) and "number_of_points" in kwargs:
        return get_point_cloud_from_triangle_mesh(mesh_or_filename=data, **kwargs)
    elif isinstance(data, str):
        return read_point_cloud(filename=data, **kwargs)
    elif isinstance(data, np.ndarray):
        return get_point_cloud_from_points(points=data, **kwargs)
    elif isinstance(data, PointCloud):
        return data
    else:
        raise TypeError(f"Can't process data of type {type(data)}.")


def scale_point_cloud(pcd, factor):
    points = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(points * factor)


def process_point_cloud(pcd,
                        voxel_size_in: float = 1.,
                        voxel_size_out: float = 1.,
                        initial_pose=np.eye(4),
                        nn_normal: int = -1,
                        nn_fpfh: int = -1,
                        fast_normal: bool = True,
                        orient_normal: int = 0,
                        visualize: bool = False):
    # assert voxel_size_in <= voxel_size_out, "Error: Output voxel size needs to greater or equal input voxel size."

    if voxel_size_in != 1.:
        scale_point_cloud(pcd, voxel_size_in)

    if voxel_size_out > voxel_size_in:
        pcd = pcd.voxel_down_sample(voxel_size_out)

    pcd.transform(initial_pose)

    if nn_normal != 0 or (nn_fpfh != 0 and not pcd.has_normals()):
        if pcd.has_normals():
            pcd.normals = o3d.utility.Vector3dVector()
        # search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        search_param = o3d.geometry.KDTreeSearchParamKNN(30 if nn_normal == -1 else nn_normal)
        pcd.estimate_normals(search_param, fast_normal)
        if orient_normal > 0:
            pcd.orient_normals_consistent_tangent_plane(orient_normal)

    if visualize:
        pcd.paint_uniform_color([50 / 255., 120 / 255., 60 / 255.])
        o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)],
                                          point_show_normal=True)

    if nn_fpfh != 0:
        # search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_out * 5, max_nn=100)
        search_param = o3d.geometry.KDTreeSearchParamKNN(10 if nn_fpfh == -1 else nn_fpfh)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)
        return pcd, pcd_fpfh
    else:
        return pcd


def read_point_cloud(filename: str, **kwargs: Any) -> PointCloud:
    """Reads point cloud data from file.

    Args:
        filename (str): The path to the triangle mesh file. 
        kwargs (any): Optional additional keyword arguments.

    Returns:
        PointCloud: The point cloud.
    """
    return o3d.io.read_point_cloud(filename=filename, **kwargs)


def read_triangle_mesh(filename: str, **kwargs: Any) -> TriangleMesh:
    """Reads triangle meshe data from file.

    Args:
        filename (str): The path to the triangle mesh file. 
        kwargs (any): Optional additional keyword arguments.

    Returns:
        TriangleMesh: The triangle mesh.
    """
    return o3d.io.read_triangle_mesh(filename, **kwargs)


def get_triangle_mesh_from_triangles_and_vertices(triangles: np.ndarray, vertices: np.ndarray) -> TriangleMesh:
    """Convenience function to obtain triangle meshes from triangles and vertices.

    Args:
        triangles (np.ndarray): triangles
        vertices (np.ndarray): vertices

    Returns:
        TriangleMesh:
    """
    triangle_mesh = TriangleMesh()
    triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    triangle_mesh.vertices = o3d.utility.Vector3Vector(vertices)
    return triangle_mesh


def get_point_cloud_from_points(points: np.ndarray) -> PointCloud:
    """Convenience function to obtain point clouds from points.

    Args:
        points (np.ndarray): A Nx3 array of points. 

    Returns:
        PointCloud: The point cloud created from the points.
    """
    point_cloud = PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


def get_point_cloud_from_triangle_mesh(mesh_or_filename: Union[TriangleMesh, str],
                                       number_of_points: int,
                                       sample_style: SampleStyle = SampleStyle.UNIFORMLY,
                                       **kwargs: Any) -> PointCloud:
    """Convenience function to obtain point clouds from triangle meshes.

    Args:
        mesh_or_filename (Union[o3d.geometry.TriangleMesh, str]): The triangle mesh from which
                                                                  the point cloud is sampled.
        number_of_points (int): number_of_points: Number of points to sample from the triangle mesh.
        sample_style (SampleStyle): sample_style: How to sample the points from the triangle mesh.
                                                  Either `UNIFORMLY` or `POISSON_DISK`. Default: `UNIFORMLY`.
        kwargs (any): Optional additional keyword arguments.

    Returns:
        PointCloud: The point cloud obtained from the tringle mesh through sampling.
    """
    if isinstance(mesh_or_filename, str):
        mesh = read_triangle_mesh(filename=mesh_or_filename, **kwargs)
    elif isinstance(mesh_or_filename, TriangleMesh):
        mesh = mesh_or_filename
    else:
        raise TypeError(f"Can't read mesh of type {type(mesh_or_filename)}.")

    if sample_style == SampleStyle.UNIFORMLY:
        return mesh.sample_points_uniformly(number_of_points=number_of_points, **kwargs)
    elif sample_style == SampleStyle.POISSON_DISK:
        return mesh.sample_points_poisson_disk(number_of_points=number_of_points, **kwargs)
    else:
        raise ValueError(f"Sample style {sample_style} not supported.")


def get_point_cloud_from_depth_image(image: Union[Image, np.ndarray], camera_intrinsics: Union[PinholeCameraIntrinsic,
                                                                                               np.ndarray],
                                     **kwargs) -> PointCloud:
    """Convenience function to obtain point clouds from depth images.

    Args:
        image (Union[Image, np.ndarray]): The depth image.
        camera_intrinsics (Union[PinholeCameraIntrinsic, np.ndarray]): The 3x3 camera intrinsics matrix.
        kwargs (any): Optional additional keyword arguments.

    Returns:
        PointCloud: The point cloud obtained from the depth image.
    """
    if isinstance(image, Image):
        _image = image
    elif isinstance(image, np.ndarray):
        _image = o3d.geometry.Image(image)
    else:
        raise TypeError(f"Image type {type(image)} not supported.")

    if isinstance(camera_intrinsics, PinholeCameraIntrinsic):
        _camera_intrinsics = camera_intrinsics
    elif isinstance(camera_intrinsics, np.ndarray):
        assert camera_intrinsics.size == 9, "Camera intrinsics must be 9 values."
        height, width = image.shape[:2]
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        _camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    else:
        raise TypeError(f"Camera intrinsics type {type(camera_intrinsics)} not supported.")

    nkwargs = {
        "depth": _image,
        "intrinsic": _camera_intrinsics,
        "extrinsic": kwargs.get("extrinsic", np.eye(4)),
        "depth_scale": kwargs.get("depth_scale", 1000.0),
        "depth_trunc": kwargs.get("depth_trunc", 1000.0),
        "stride": kwargs.get("stride", 1),
        "project_valid_depth_only": kwargs.get("project_valid_depth_only", True)
    }
    return o3d.geometry.PointCloud.create_from_depth_image(**nkwargs)
