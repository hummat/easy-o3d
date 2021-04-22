"""Utility functions used throughout the project.

Classes:
    utils.ValueTypes: Flags for default values.
    utils.SampleTypes: Sample style flags.
    utils.DownsampleTypes: Downsample style flags.
    utils.SearchParamTypes: KNN and FPFH feature computation style flags.
    utils.OrientationTypes: Normal orientation style flags.

Functions:
    utils.eval_data: Automatically determines the data type and loads the data accordingly.
    utils.process_point_cloud: Implements various point cloud processing functionality.
    utils.read_point_cloud: Reads point cloud data from file.
    utils.read_triangle_mesh: Reads triangle mesh data from file.
    utils.read_triangle_mesh_from_triangles_and_vertices: Constructs triangle mesh from triangles and vertices.
    utils.get_point_cloud_from_points: Constructs point cloud from list or array of points.
    utils.sample_point_cloud_from_triangle_mesh: Samples point clouds from triangle meshes.
    utils.get_camera_intrinsic_from_array: Construct camera intrinsic object from array.
    utils.get_rgbd_image: Constructs RGB-D image from color and depth.
    utils.eval_image_type: Constructs RGB or RGB-D image depending on input.
    utils.eval_camera_intrinsic_type: Construct camera intrinsic object from input.
    utils.convert_depth_image_to_point_cloud: Converts depth image to point cloud.
    utils.convert_rgbd_image_to_point_cloud: Converts RGB-D image to colored point cloud.
    utils.get_transformation_matrix_from_xyz: Returns 4x4 transformation matrix from Euler XYZ rotation and translation.
    utils.get_camera_parameters_from_blenderproc_bopwriter: Returns intrinsic and extrinsic camera parameters from them
                                                            output of the BopWriter implemented in BlenderProc.
"""

import copy
import json
import logging
import os
from enum import Flag, auto
from typing import Any, List, Union, Tuple

import numpy as np
import open3d as o3d

PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic
PinholeCameraIntrinsicParameters = o3d.camera.PinholeCameraIntrinsicParameters
Image = o3d.geometry.Image
RGBDImage = o3d.geometry.RGBDImage
PointCloud = o3d.geometry.PointCloud
TriangleMesh = o3d.geometry.TriangleMesh
Feature = o3d.pipelines.registration.Feature


class ValueTypes(Flag):
    DEFAULT = -1


class SampleTypes(Flag):
    UNIFORMLY = auto()
    POISSON_DISK = auto()


class DownsampleTypes(Flag):
    NONE = auto()
    VOXEL = auto()
    RANDOM = auto()
    UNIFORM = auto()


class SearchParamTypes(Flag):
    KNN = auto()
    RADIUS = auto()
    HYBRID = auto()


class OrientationTypes(Flag):
    NONE = auto()
    TANGENT_PLANE = auto()
    CAMERA = auto()
    DIRECTION = auto()


InputTypes = Union[str, np.ndarray, PointCloud, TriangleMesh, Image, RGBDImage]
ImageTypes = Union[Image, RGBDImage, np.ndarray, str]
RGBDImageTypes = Union[ImageTypes, List[ImageTypes]]
CameraTypes = Union[np.ndarray, PinholeCameraIntrinsic, PinholeCameraIntrinsicParameters]

logger = logging.getLogger(__name__)


def eval_data(data: InputTypes, **kwargs: Any) -> PointCloud:
    """Convenience function that automatically determines the data type and loads the data accordingly.

    Args:
        data (InputTypes): The data to be evaluated (read, transformed).
        kwargs (any): Optional additional keyword arguments. If `camera_intrinsic` is contained,
                      loads from `Image` or `RGBDImage` depending on `data.shape`.
                      If `number_of_points` is contained, loads and samples from `TriangleMesh`.

    Returns:
        PointCloud: The transformed point cloud data.
    """
    if isinstance(data, PointCloud):
        return data
    elif isinstance(data, Image) and "camera_intrinsic" in kwargs:
        return convert_depth_image_to_point_cloud(depth_image=data, **kwargs)
    elif isinstance(data, (RGBDImage, list)) and "camera_intrinsic" in kwargs:
        return convert_rgbd_image_to_point_cloud(rgbd_image=data, **kwargs)
    elif isinstance(data, (str, TriangleMesh)) and "number_of_points" in kwargs:
        return sample_point_cloud_from_triangle_mesh(mesh_or_filename=data, **kwargs)
    elif isinstance(data, str):
        return read_point_cloud(filename=data, **kwargs)
    elif isinstance(data, np.ndarray):
        if "camera_intrinsic" in kwargs:
            if len(data.shape) == 2:
                return convert_depth_image_to_point_cloud(depth_image=data, **kwargs)
            elif len(data.shape) == 4:
                return convert_rgbd_image_to_point_cloud(rgbd_image=data, **kwargs)
        elif len(data.shape) in [3, 6, 9]:
            return get_point_cloud_from_points(points=data)
        else:
            raise ValueError(
                f"Point cloud data must be of shape Nx3 (xyz), Nx6 or Nx9 (rgb, normals) but is {data.shape}.")
    else:
        raise TypeError(f"Can't process data of type {type(data)} with keyword arguments {kwargs}.")


def process_point_cloud(point_cloud: PointCloud,
                        downsample: DownsampleTypes = DownsampleTypes.NONE,
                        downsample_factor: Union[float, int] = 1.0,
                        transform: np.ndarray = None,
                        scale: float = 1.0,
                        estimate_normals: bool = False,
                        fast_normal_computation: bool = True,
                        normalize_normals: bool = False,
                        orient_normals: OrientationTypes = OrientationTypes.NONE,
                        recalculate_normals: bool = False,
                        compute_feature: bool = False,
                        search_param: SearchParamTypes = SearchParamTypes.HYBRID,
                        search_param_knn: int = 30,
                        search_param_radius: float = 0.02,  # 2cm
                        camera_location_or_direction: np.ndarray = np.zeros(3),
                        draw: bool = False) -> Union[PointCloud, Tuple[PointCloud, Feature]]:

    _point_cloud = copy.deepcopy(point_cloud)
    if downsample != DownsampleTypes.NONE:
        logger.debug(f"{downsample} downsampling point cloud  with factor {downsample_factor}.")
        logger.debug(f"Number of points before downsampling: {len(np.asarray(_point_cloud.points))}")
        if downsample == DownsampleTypes.VOXEL:
            _point_cloud = _point_cloud.voxel_down_sample(voxel_size=downsample_factor)
        elif downsample == DownsampleTypes.RANDOM:
            raise NotImplementedError
            # _point_cloud = _point_cloud.random_down_sample(sampling_ratio=downsample_factor)
        elif downsample == DownsampleTypes.UNIFORM:
            _point_cloud = _point_cloud.uniform_down_sample(every_k_points=int(downsample_factor))
        else:
            raise TypeError(f"`downsample` needs to by of type {DownsampleTypes} but is {type(downsample)}.")
        logger.debug(f"Number of points after downsampling: {len(np.asarray(_point_cloud.points))}")

    if transform is not None:
        _transform = np.asarray(transform)
        if _transform.size in [3, 4]:
            _point_cloud.translate(translation=_transform.ravel()[:3], relative=True)
        elif _transform.size == 9:
            _point_cloud.rotate(R=_transform.reshape(3, 3), center=_point_cloud.get_center())
        elif _transform.size == 16:
            _point_cloud.transform(_transform.reshape(4, 4))
        else:
            raise ValueError("`transform` needs to be a valid translation, rotation or transformation in natural or" \
                             "homogeneous coordinates, i.e. of size 3, 4, 9 or 16.")

    if scale != 1.0:
        _point_cloud.scale(scale=scale, center=_point_cloud.get_center())

    if estimate_normals or compute_feature:
        if search_param == SearchParamTypes.KNN:
            _search_param = o3d.geometry.KDTreeSearchParamKNN(knn=search_param_knn)
        elif search_param == SearchParamTypes.RADIUS:
            _search_param = o3d.geometry.KDTreeSearchParamRadius(radius=search_param_radius)
        elif search_param == SearchParamTypes.HYBRID:
            _search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=search_param_radius, max_nn=search_param_knn)
        else:
            raise TypeError(f"`search_param` needs to be of type {SearchParamTypes} but is {type(search_param)}.")

    if estimate_normals and (not _point_cloud.has_normals() or recalculate_normals):
        logger.debug(f"Estimating point cloud normals using method {search_param}.")
        if recalculate_normals:
            _point_cloud.normals = o3d.utility.Vector3dVector()
        _point_cloud.estimate_normals(search_param=_search_param, fast_normal_computation=fast_normal_computation)

    if normalize_normals:
        if _point_cloud.has_normals():
            _point_cloud = _point_cloud.normalize_normals()
        else:
            logger.warning("Point cloud doesnt' have normals so can't normalize them.")

    assert isinstance(orient_normals, OrientationTypes)
    if orient_normals != OrientationTypes.NONE:
        assert _point_cloud.has_normals(), "Point cloud doesn't have normals so can't orient them."
        logger.debug(f"Orienting normals towards {orient_normals}.")
    if orient_normals == OrientationTypes.TANGENT_PLANE:
        _point_cloud.orient_normals_consistent_tangent_plane(k=search_param_knn)
    elif orient_normals == OrientationTypes.CAMERA:
        _point_cloud.orient_normals_towards_camera_location(camera_location=camera_location_or_direction)
    elif orient_normals == OrientationTypes.DIRECTION:
        _point_cloud.orient_normals_to_align_with_direction(orientation_reference=camera_location_or_direction)

    if compute_feature:
        feature = o3d.pipelines.registration.compute_fpfh_feature(input=_point_cloud, search_param=_search_param)

    if draw:
        if not _point_cloud.has_colors():
            _point_cloud.paint_uniform_color([0.8, 0.0, 0.0])
        o3d.visualization.draw_geometries([_point_cloud], window_name="Processed Point Cloud", width=800, height=600)

    if compute_feature:
        return _point_cloud, feature
    else:
        return _point_cloud


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
    triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
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


def sample_point_cloud_from_triangle_mesh(mesh_or_filename: Union[TriangleMesh, str],
                                          number_of_points: int,
                                          sample_type: SampleTypes = SampleTypes.UNIFORMLY,
                                          **kwargs: Any) -> PointCloud:
    """Convenience function to obtain point clouds from triangle meshes.

    Args:
        mesh_or_filename (Union[o3d.geometry.TriangleMesh, str]): The triangle mesh from which
                                                                  the point cloud is sampled.
        number_of_points (int): Number of points to sample from the triangle mesh.
        sample_type (SampleTypes): How to sample the points from the triangle mesh.
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

    if sample_type == SampleTypes.UNIFORMLY:
        return mesh.sample_points_uniformly(number_of_points=number_of_points, **kwargs)
    elif sample_type == SampleTypes.POISSON_DISK:
        return mesh.sample_points_poisson_disk(number_of_points=number_of_points, **kwargs)
    else:
        raise ValueError(f"Sample style {sample_type} not supported.")


def get_camera_intrinsic_from_array(image_or_path: ImageTypes, camera_intrinsic: Union[np.ndarray,
                                                                                       list]) -> PinholeCameraIntrinsic:
    assert camera_intrinsic.size == 9, "Camera intrinsic must be 9 values but are {camera_intrinsic.size}."
    intrinsic = camera_intrinsic
    if isinstance(intrinsic, list):
        intrinsic = np.asarray(camera_intrinsic)
    intrinsic = intrinsic.reshape(3, 3)
    image = eval_image_type(image_or_path=image_or_path)
    if isinstance(image, RGBDImage):
        image = image.depth
    height, width = np.asarray(image).shape[:2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    return PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)


def get_rgbd_image(color: ImageTypes, depth: Union[ImageTypes, None] = None, **kwargs: Any) -> RGBDImage:
    if depth is None and isinstance(color, RGBDImage):
        return color

    if isinstance(color, str):
        _color = np.asarray(o3d.io.read_image(color))
    elif isinstance(color, Image):
        _color = np.asarray(color)
    elif isinstance(color, np.ndarray):
        _color = color

    if isinstance(depth, str):
        _depth = np.asarray(o3d.io.read_image(depth))
    elif isinstance(depth, Image):
        _depth = np.asarray(depth)
    elif isinstance(depth, np.ndarray):
        _depth = depth

    if _depth is None:
        assert len(_color.shape) == 4, "Color image must have depth channel if depth is not provided."
        _color = Image(_color[:, :, :3])
        _depth = Image(_color[:, :, 3])
        return RGBDImage.create_from_color_and_depth(color=_color, depth=_depth, **kwargs)
    else:
        return RGBDImage.create_from_color_and_depth(color=Image(_color), depth=Image(_depth), **kwargs)


def eval_image_type(image_or_path: RGBDImageTypes, **kwargs: Any) -> Union[Image, RGBDImage]:
    if isinstance(image_or_path, (Image, RGBDImage)):
        return image_or_path

    if isinstance(image_or_path, list):
        assert len(image_or_path) == 2, "Need to provide exactly one color and one depth image."
        color = image_or_path[0]
        depth = image_or_path[1]
        return get_rgbd_image(color=color, depth=depth, **kwargs)
    else:
        if isinstance(image_or_path, str):
            color_or_depth = np.asarray(o3d.io.read_image(image_or_path))
        elif isinstance(image_or_path, np.ndarray):
            color_or_depth = image_or_path

        if len(color_or_depth.shape) in [2, 3]:
            return Image(image_or_path)
        elif len(color_or_depth.shape) == 4:
            return get_rgbd_image(color=image_or_path, **kwargs)
        else:
            raise TypeError(f"Image type {type(image_or_path)} not supported.")


def eval_camera_intrinsic_type(image_or_path: ImageTypes, camera_intrinsic: CameraTypes):
    if isinstance(camera_intrinsic, (PinholeCameraIntrinsic, PinholeCameraIntrinsicParameters)):
        return camera_intrinsic
    elif isinstance(camera_intrinsic, (np.ndarray, list)):
        return get_camera_intrinsic_from_array(image_or_path=image_or_path, camera_intrinsic=camera_intrinsic)
    else:
        raise TypeError(f"Camera intrinsic type {type(camera_intrinsic)} not supported.")


def convert_depth_image_to_point_cloud(image_or_path: ImageTypes,
                                       camera_intrinsic: CameraTypes,
                                       camera_extrinsic: Union[np.ndarray, list] = np.eye(4),
                                       depth_scale: float = 1000.0,
                                       depth_trunc: float = 1000.0,
                                       **kwargs: Any) -> PointCloud:
    """Convenience function to obtain point clouds from depth images.

    Args:
        image (Union[Image, np.ndarray]): The depth image.
        camera_intrinsic (Union[PinholeCameraIntrinsic,
                                 PinholeCameraIntrinsicParameters,
                                 np.ndarray]): The 3x3 camera intrinsic matrix.
        kwargs (any): Optional additional keyword arguments.

    Returns:
        PointCloud: The point cloud obtained from the depth image.
    """

    image = eval_image_type(image_or_path=image_or_path, **kwargs)
    assert isinstance(image, Image)
    assert len(np.asarray(image).shape) == 2, f"Depth image must have shape WxH but is {image.shape}."
    intrinsic = eval_camera_intrinsic_type(image_or_path=image, camera_intrinsic=camera_intrinsic)
    extrinsic = np.asarray(camera_extrinsic).reshape(4, 4)

    nkwargs = {
        "depth": image,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "depth_scale": depth_scale,
        "depth_trunc": depth_trunc,
        "stride": kwargs.get("stride", 1),
        "project_valid_depth_only": kwargs.get("project_valid_depth_only", True)
    }
    return PointCloud.create_from_depth_image(**nkwargs)


def convert_rgbd_image_to_point_cloud(rgbd_image_or_path: RGBDImageTypes,
                                      camera_intrinsic: CameraTypes,
                                      camera_extrinsic: Union[np.ndarray, list] = np.eye(4),
                                      depth_scale: float = 1000.0,
                                      depth_trunc: float = 1000.0,
                                      **kwargs: Any) -> PointCloud:
    nkwargs = {
        "depth_scale": depth_scale,
        "depth_trunc": depth_trunc,
        "convert_rgb_to_intensity": kwargs.get("convert_rgb_to_intensity", False),
    }
    rgbd_image = eval_image_type(image_or_path=rgbd_image_or_path, **nkwargs)
    assert isinstance(rgbd_image, RGBDImage)

    intrinsic = eval_camera_intrinsic_type(image_or_path=rgbd_image, camera_intrinsic=camera_intrinsic)
    extrinsic = np.asarray(camera_extrinsic).reshape(4, 4)
    nkwargs = {
        "image": rgbd_image,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "project_valid_depth_only": kwargs.get("project_valid_depth_only", True)
    }
    return PointCloud.create_from_rgbd_image(**nkwargs)


def get_transformation_matrix_from_xyz(rotation_xyz: Union[np.array, list] = np.zeros(3),
                                       translation_xyz: Union[np.array, list] = np.zeros(3)) -> np.ndarray:
    rx, ry, rz = np.asarray(rotation_xyz).ravel()[:3]
    tx, ty, tz = np.asarray(translation_xyz).ravel()[:3]
    R = PointCloud.get_rotation_matrix_from_xyz(np.array([np.radians(rx), np.radians(ry), np.radians(rz)]))
    t = [tx, ty, tz]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def get_transformation_matrix_from_quaternion(rotation_wxyz: Union[np.array, list] = np.zeros(4),
                                              translation_xyz: Union[np.array, list] = np.zeros(3)) -> np.ndarray:
    rotation = np.asarray(rotation_wxyz).ravel()[:4]
    tx, ty, tz = np.asarray(translation_xyz).ravel()[:3]
    R = PointCloud.get_rotation_matrix_from_quaternion(rotation=rotation)
    t = [tx, ty, tz]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def get_camera_parameters_from_blenderproc_bopwriter(scene_id: Union[int, str], path_to_scene_camera_json: str,
                                                     path_to_camera_json,
                                                     output_path: None) -> o3d.camera.PinholeCameraParameters:
    scene_id = str(scene_id)

    # Get camera extrinsic
    with open(path_to_scene_camera_json) as f:
        extrinsic_data = json.load(f)

    R_w2c = np.asarray(extrinsic_data[scene_id]['cam_R_w2c']).reshape(3, 3)
    t_w2c = np.asarray(extrinsic_data[scene_id]['cam_t_w2c']) / 1000.0
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = t_w2c
    extrinsic = T_w2c

    # Get camera intrinsic
    with open(path_to_camera_json) as f:
        intrinsic_data = json.load(f)

    width, height = intrinsic_data['width'], intrinsic_data['height']
    cx, cy = intrinsic_data['cx'], intrinsic_data['cy']
    fx, fy = intrinsic_data['fx'], intrinsic_data['fy']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    # Set camera parameters
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic = intrinsic
    camera_parameters.extrinsic = extrinsic

    # Maybe write to file
    if output_path is not None:
        o3d.io.write_pinhole_camera_parameters(os.path.join(output_path, "camera_parameters.json"), camera_parameters)

    return camera_parameters


def draw_geometries(geometries: list, window_name="Visualizer", size=(800, 600)):
    o3d.visualization.draw_geometries(geometries, window_name=window_name, width=size[0], height=size[1])
