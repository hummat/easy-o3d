"""Utility functions used throughout the project.

Classes:
    SampleTypes: Supported types of point cloud sampling from meshes.
    DownsampleTypes: Supported point cloud downsampling types.
    OutlierTypes: Supported outlier removal types.
    SearchParamTypes: Supported normal and FPFH feature computation search parameter types.
    OrientationTypes: Supported normal orientation types.

Functions:
    eval_data: Convenience function that automatically determines the data type and loads the data accordingly.
    eval_data_parallel: Evaluates a list of inputs in parallel using multi-threading.
    process_point_cloud: Utility function to apply various processing steps on point cloud data.
    process_point_cloud_parallel: Processes a list of point clouds in parallel using multi-threading.
    read_point_cloud: Reads point cloud data from file.
    read_triangle_mesh: Reads triangle mesh data from file.
    get_triangle_mesh_from_triangles_and_vertices: Convenience function to obtain triangle meshes from triangles and
                                                   vertices.
    get_point_cloud_from_points: Convenience function to obtain point clouds from points.
    sample_point_cloud_from_triangle_mesh: Convenience function to obtain point clouds from triangle meshes.
    get_camera_intrinsic_from_array: Constructs camera intrinsic object from image dimensions and camara intrinsic data.
    get_rgbd_image: Constructs an RGB-D image from a color and a depth image.
    eval_image_type: Convenience function constructing an RGB or RGB-D image based on input type.
    eval_camera_intrinsic_type: Convenience function constructing a camera intrinsic object based on input type.
    convert_depth_image_to_point_cloud: Convenience function converting depth images to point clouds.
    convert_rgbd_image_to_point_cloud: Convenience function converting RGB-D image data to point clouds.
    eval_transformation_data: Evaluates different types of transformation data to obtain a 4x4 transformation matrix.
    get_transformation_matrix_from_xyz: Constructs a 4x4 homogenous transformation matrix from a XYZ translation vector
                                        and XYZ Euler angles.
    get_transformation_matrix_from_quaternion: Constructs a 4x4 homogenous transformation matrix from a XYZ translation
                                               vector and WXYZ quaternion values.
    get_ground_truth_pose_from_file: Reads ground truth from JSON file.
    get_camera_intrinsic_from_blenderproc_bopwriter: Constructs intrinsic camera parameter object from BlenderProc
                                                     BopWriter data.
    get_camera_extrinsic_from_blenderproc_bopwriter: Reads camera extrinsic transform from BlenderProc Bopwriter data.
    get_camera_parameters_from_blenderproc_bopwriter: Constructs intrinsic and extrinsic camera parameter object from
                                                      BlenderProc BopWriter data.
    read_camera_parameters: Reads pinhole camera parameters (intrinsic, extrinsic) from file.
    read_camera_intrinsic: Reads pinhole camera intrinsic parameters from file.
    draw_geometries: Convenience function to draw 3D geometries.
    get_transformation_error: Computes the rotational and translational error between estimated and ground-truth
                              transformation data.
    get_rotation_error: Computes the error between estimated and ground-truth rotation in degrees or radians.
    get_translation_error: Computes the mean-squared translational error between estimated and ground-truth translation.
"""
import copy
import json
import logging
import os
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from enum import Flag, auto
from typing import Any, List, Union, Tuple, Dict
import math

import numpy as np
import open3d as o3d

PinholeCameraParameters = o3d.camera.PinholeCameraParameters
PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic
PinholeCameraIntrinsicParameters = o3d.camera.PinholeCameraIntrinsicParameters
Image = o3d.geometry.Image
RGBDImage = o3d.geometry.RGBDImage
PointCloud = o3d.geometry.PointCloud
TriangleMesh = o3d.geometry.TriangleMesh
Feature = o3d.pipelines.registration.Feature

ImageTypes = Union[Image, RGBDImage, np.ndarray, str]
RGBDImageTypes = Union[ImageTypes, List[ImageTypes]]
InputTypes = Union[PointCloud, TriangleMesh, RGBDImageTypes]
CameraTypes = Union[list, np.ndarray, PinholeCameraIntrinsic, PinholeCameraIntrinsicParameters, str]
TransformationTypes = Union[np.ndarray, List[float], List[List[float]], str]

logger = logging.getLogger(__name__)


class SampleTypes(Flag):
    """Supported types of point cloud sampling from meshes."""
    UNIFORMLY = auto()
    POISSON_DISK = auto()


class DownsampleTypes(Flag):
    """Supported point cloud downsampling types."""
    VOXEL = auto()
    RANDOM = auto()
    UNIFORM = auto()


class OutlierTypes(Flag):
    """Supported outlier removal types."""
    STATISTICAL = auto()
    RADIUS = auto()


class SearchParamTypes(Flag):
    """Supported normal and FPFH feature computation search parameter types."""
    KNN = auto()
    RADIUS = auto()
    HYBRID = auto()


class OrientationTypes(Flag):
    """Supported normal orientation types."""
    TANGENT_PLANE = auto()
    CAMERA = auto()
    DIRECTION = auto()


def eval_data(data: InputTypes,
              number_of_points: Union[int, None] = None,
              camera_intrinsic: Union[np.ndarray, list, None] = None,
              **kwargs: Any) -> PointCloud:
    """Convenience function that automatically determines the data type and loads the data accordingly.

    Args:
        data: The data to be evaluated (read, transformed).
        number_of_points: Number of points to sample from `data` if it is a triangle mesh.
        camera_intrinsic: Camera intrinsic parameters used when converting depth or RGB-D images to point clouds.

    Returns:
        The data evaluated as a point cloud.
    """
    if isinstance(data, PointCloud):
        logger.debug("Data is point cloud. Returning.")
        return data
    elif isinstance(data, (Image, str)) and camera_intrinsic is not None:
        logger.debug("Trying to convert depth image to point cloud.")
        return convert_depth_image_to_point_cloud(image_or_path=data,
                                                  camera_intrinsic=camera_intrinsic,
                                                  **kwargs)
    elif isinstance(data, (RGBDImage, list)) and camera_intrinsic is not None:
        logger.debug("Trying to convert RGB-D image to point cloud.")
        return convert_rgbd_image_to_point_cloud(rgbd_image_or_path=data,
                                                 camera_intrinsic=camera_intrinsic,
                                                 convert_rgb_to_intensity=kwargs.get("convert_rgb_to_intensity", False),
                                                 **kwargs)
    elif isinstance(data, (TriangleMesh, str)) and number_of_points is not None:
        logger.debug("Trying to sample point cloud from mesh.")
        return sample_point_cloud_from_triangle_mesh(mesh_or_filename=data,
                                                     number_of_points=number_of_points,
                                                     **kwargs)
    elif isinstance(data, str):
        logger.debug(f"Trying to read point cloud data from file.")
        return read_point_cloud(filename=data)
    elif isinstance(data, np.ndarray):
        if camera_intrinsic is not None:
            if len(data.shape) == 2:
                logger.debug(f"Trying to convert depth data to point cloud ")
                return convert_depth_image_to_point_cloud(image_or_path=data,
                                                          camera_intrinsic=camera_intrinsic,
                                                          **kwargs)
            elif data.shape[2] == 4:
                logger.debug(f"Trying to convert RGB-D data to point cloud.")
                return convert_rgbd_image_to_point_cloud(rgbd_image_or_path=data,
                                                         camera_intrinsic=camera_intrinsic,
                                                         **kwargs)
        elif data.shape[1] in [3, 6, 9]:
            logger.debug("Trying to convert data to point cloud.")
            return get_point_cloud_from_points(points=data)
        else:
            raise ValueError(
                f"Point cloud data must be of shape Nx3 (xyz), Nx6 or Nx9 (rgb, normals) but is {data.shape}.")
    else:
        raise TypeError(f"Can't process data of type {type(data)}.")


def eval_data_parallel(data_list: List[InputTypes],
                       num_threads: int = cpu_count(),
                       **kwargs: Any) -> List[PointCloud]:
    """Evaluates a list of inputs in parallel using multi-threading.

    Args:
        data_list: The list of inputs.
        num_threads: The number of parallel threads to run.

    Returns:
        List of evaluated inputs.
    """
    kwargs_list = list()
    for i, d in enumerate(data_list):
        kwargs_dict = {"data": d}
        for key, value in kwargs.items():
            kwargs_dict[key] = value[i] if isinstance(value, list) else value
        kwargs_list.append(kwargs_dict)
    if len(data_list) == 1:
        return [eval_data(**kwargs_list[0])]
    parallel = Parallel(n_jobs=min(num_threads, len(data_list)), prefer="threads")
    return parallel(delayed(eval_data)(**params) for params in kwargs_list)


def process_point_cloud(point_cloud: PointCloud,
                        scale: float = 1.0,
                        downsample: Union[DownsampleTypes, None] = None,
                        downsample_factor: Union[float, int] = 1,
                        remove_outlier: Union[OutlierTypes, None] = None,
                        outlier_std_ratio: float = 1.0,
                        transformation: [np.ndarray, list, None] = None,
                        estimate_normals: bool = False,
                        recalculate_normals: bool = False,
                        fast_normal_computation: bool = True,
                        normalize_normals: bool = False,
                        orient_normals: Union[OrientationTypes, None] = None,
                        compute_feature: bool = False,
                        search_param: Union[SearchParamTypes, None] = SearchParamTypes.HYBRID,
                        search_param_knn: int = 30,
                        search_param_radius: float = 0.02,  # 2cm
                        camera_location_or_direction: [np.ndarray, list] = np.zeros(3),
                        draw: bool = False) -> Union[PointCloud, Tuple[PointCloud, Feature]]:
    """Utility function to apply various processing steps on point cloud data.

    Processing steps are applied in order implied by the functions argument order:
    1. `scale`
    2. `downsample`
    3. `remove outlier`
    4. `transformation`
    5. `estimate normals`
    6. `normalize normals`
    7. `orient normals`
    8. `compute feature`

    To estimate normals and compute feature with individual search parameters, rerun the function for each step.

    Args:
        point_cloud: The point cloud to be processed.
        scale: Scales the point cloud.
        downsample: Reduce point cloud density by dropping points uniformly or in voxel grid fashion.
        downsample_factor: The amount of downsampling. Factor for `DownsampleType.UNIFORM`, voxel size for
                           `DownsampleType.VOXEL`.
        remove_outlier: Remove outlier vertices based on radius density or variance.
        outlier_std_ratio: Standard deviation for statistical outlier removal. Smaller removes more vertices.
        transformation: Homogeneous transformation. Also accepts translation vector or rotation matrix.
        estimate_normals: Estimate vertex normals.
        recalculate_normals: Recalculate normals if the point cloud already has normals.
        fast_normal_computation: Use fast normal computation algorithm.
        normalize_normals: Scale normals to unit length.
        orient_normals: Orient normals towards: plane spanned by their neighbors, an orientation or the camera location.
        compute_feature: Compute FPFH feature for global registration algorithms.
        search_param: Normal and FPFH feature computation search parameters. Can be radius, kNN or both (hybrid).
        search_param_knn: Compute normals and FPFH features based on k neighboring vertices.
        search_param_radius: Compute normals and FPFH features based on vertices inside a specified radius.
        camera_location_or_direction: The camera location or an orientation used in normal orientation computation.
        draw: Visualize the processed point cloud. Mostly for debugging.

    Returns:
        The processed point cloud.
    """
    start = time.time()
    _point_cloud = copy.deepcopy(point_cloud)
    if scale != 1.0:
        logger.debug(f"Scaling point cloud with factor {scale}.")
        logger.debug("Using custom scaling code as PointCloud.scale doesn't seem to work.")
        _point_cloud.points = o3d.utility.Vector3dVector(np.asarray(_point_cloud.points) * scale)
        # FIXME: Open3D factory `scale` function doesn't do anything.
        # _point_cloud = _point_cloud.scale(scale=scale, center=_point_cloud.get_center())

    if downsample is not None:
        logger.debug(f"{downsample} downsampling point cloud with factor {downsample_factor}.")
        logger.debug(f"Number of points before downsampling: {len(np.asarray(_point_cloud.points))}")
        if downsample == DownsampleTypes.VOXEL:
            _point_cloud = _point_cloud.voxel_down_sample(voxel_size=downsample_factor)
        elif downsample == DownsampleTypes.RANDOM:
            raise NotImplementedError
            # FIXME: Open3D `random_down_sample` is in docu but not in code.
            # _point_cloud = _point_cloud.random_down_sample(sampling_ratio=downsample_factor)
        elif downsample == DownsampleTypes.UNIFORM:
            _point_cloud = _point_cloud.uniform_down_sample(every_k_points=int(downsample_factor))
        else:
            raise ValueError(f"`downsample` needs to by one of `DownsampleTypes` but is {type(downsample)}.")
        logger.debug(f"Number of points after downsampling: {len(np.asarray(_point_cloud.points))}")

    if remove_outlier is not None:
        num_points = len(np.asarray(_point_cloud.points))
        if remove_outlier == OutlierTypes.STATISTICAL:
            _point_cloud, _ = _point_cloud.remove_statistical_outlier(nb_neighbors=search_param_knn,
                                                                      std_ratio=outlier_std_ratio)
        elif remove_outlier == OutlierTypes.RADIUS:
            _point_cloud, _ = _point_cloud.remove_radius_outlier(nb_points=search_param_knn, radius=search_param_radius)
        else:
            raise ValueError(f"`remove_outlier` needs to be one of `OutlierTypes` but is {type(remove_outlier)}.")
        logger.debug(f"Removed {num_points - len(np.asarray(_point_cloud.points))} outliers.")

    if transformation is not None:
        _transform = np.asarray(transformation)
        if _transform.size in [3, 4]:
            _point_cloud.translate(translation=_transform.ravel()[:3], relative=True)
        elif _transform.size == 9:
            # noinspection PyArgumentList
            _point_cloud.rotate(R=_transform.reshape(3, 3), center=_point_cloud.get_center())
        elif _transform.size == 16:
            _point_cloud.transform(_transform.reshape(4, 4))
        else:
            raise ValueError("`transformation` needs to be a valid translation, rotation or transformation in natural"
                             "or homogeneous coordinates, i.e. of size 3, 4, 9 or 16.")

    if (estimate_normals or compute_feature) and search_param is not None:
        if search_param == SearchParamTypes.KNN:
            _search_param = o3d.geometry.KDTreeSearchParamKNN(knn=search_param_knn)
        elif search_param == SearchParamTypes.RADIUS:
            _search_param = o3d.geometry.KDTreeSearchParamRadius(radius=search_param_radius)
        elif search_param == SearchParamTypes.HYBRID:
            _search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=search_param_radius, max_nn=search_param_knn)
        else:
            raise TypeError(f"`search_param` needs have type `SearchParamTypes` but has type {type(search_param)}.")
    else:
        _search_param = None

    if estimate_normals and (not _point_cloud.has_normals() or recalculate_normals) and _search_param is not None:
        logger.debug(f"Estimating point cloud normals using method {search_param}.")
        if recalculate_normals:
            _point_cloud.normals = o3d.utility.Vector3dVector()
        _point_cloud.estimate_normals(search_param=_search_param, fast_normal_computation=fast_normal_computation)

    if normalize_normals:
        if _point_cloud.has_normals():
            _point_cloud = _point_cloud.normalize_normals()
        else:
            logger.warning("Point cloud doesnt' have normals so can't normalize them.")

    if orient_normals is not None:
        assert _point_cloud.has_normals(), "Point cloud doesn't have normals which could be oriented."
        logger.debug(f"Orienting normals towards {orient_normals}.")
        if orient_normals == OrientationTypes.TANGENT_PLANE:
            _point_cloud.orient_normals_consistent_tangent_plane(k=search_param_knn)
        elif orient_normals == OrientationTypes.CAMERA:
            _point_cloud.orient_normals_towards_camera_location(camera_location=camera_location_or_direction)
        elif orient_normals == OrientationTypes.DIRECTION:
            _point_cloud.orient_normals_to_align_with_direction(
                orientation_reference=np.asarray(camera_location_or_direction))
        else:
            raise ValueError(f"`orient_normals` needs to be one of `OrientationTypes` but is {type(orient_normals)}.")

    if compute_feature:
        assert _point_cloud.has_normals(), "Point cloud doesn't have normals which are needed to compute FPFH feature."
        logger.debug(f"Computing FPFH features using method {search_param}.")
        # noinspection PyTypeChecker
        feature = o3d.pipelines.registration.compute_fpfh_feature(input=_point_cloud, search_param=_search_param)
    else:
        feature = None

    logger.debug(f"Processing took {time.time() - start} seconds.")

    if draw:
        if not _point_cloud.has_colors():
            _point_cloud.paint_uniform_color([0.8, 0.0, 0.0])
        draw_geometries(geometries=[_point_cloud], window_name="Processed Point Cloud")

    if compute_feature and feature:
        return _point_cloud, feature
    else:
        return _point_cloud


def process_point_cloud_parallel(point_cloud_list: List[PointCloud],
                                 num_threads: int = cpu_count(),
                                 **kwargs: Any) -> List[PointCloud]:
    """Processes a list of point clouds in parallel using multi-threading.

    Args:
        point_cloud_list: A list of point clouds to process.
        num_threads: The number of parallel threads to run.

    Returns:
        A list of processed point clouds.
    """
    kwargs_list = list()
    for i, point_cloud in enumerate(point_cloud_list):
        kwargs_dict = {"point_cloud": point_cloud}
        for key, value in kwargs.items():
            kwargs_dict[key] = value[i] if isinstance(value, list) else value
        kwargs_list.append(kwargs_dict)
    if len(point_cloud_list) == 1:
        return [process_point_cloud(**kwargs_list[0])]
    parallel = Parallel(n_jobs=min(num_threads, len(point_cloud_list)), prefer="threads")
    return parallel(delayed(process_point_cloud)(**params) for params in kwargs_list)


def read_point_cloud(filename: str, **kwargs: Any) -> PointCloud:
    """Reads point cloud data from file.

    Args:
        filename: The path to the triangle mesh file.

    Returns:
        The point cloud data read from file.
    """
    if filename.endswith(".np") or filename.endswith(".npz"):
        potential_points = np.load(filename)
        if potential_points.shape[1] in [3, 6, 9]:
            return get_point_cloud_from_points(points=potential_points)
        else:
            raise ValueError(f"Numpy array read from file has shape {potential_points.shape} which is not supported.")
    return o3d.io.read_point_cloud(filename=filename,
                                   format=kwargs.get("format", 'auto'),
                                   remove_nan_points=kwargs.get("remove_nan_points", True),
                                   remove_infinite_points=kwargs.get("remove_infinite_points", True),
                                   print_progress=kwargs.get("print_progress", False))


def read_triangle_mesh(filename: str, **kwargs: Any) -> TriangleMesh:
    """Reads triangle mesh data from file.

    Args:
        filename: The path to the triangle mesh file.

    Returns:
        The triangle mesh data read from file.
    """
    return o3d.io.read_triangle_mesh(filename=filename,
                                     enable_post_processing=kwargs.get("enable_post_processing", False),
                                     print_progress=kwargs.get("print_progress", False))


def get_triangle_mesh_from_triangles_and_vertices(triangles: Union[np.ndarray, list],
                                                  vertices: Union[np.ndarray, list]) -> TriangleMesh:
    """Convenience function to obtain triangle meshes from triangles and vertices.

    Args:
        triangles: The mesh triangle indices.
        vertices: The mesh vertex coordinates.

    Returns:
        The triangle mesh constructed from triangles and vertices.
    """
    triangle_mesh = TriangleMesh()
    triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return triangle_mesh


def get_point_cloud_from_points(points: np.ndarray) -> PointCloud:
    """Convenience function to obtain point clouds from points.

    Args:
        points: A Nx3 array of vertex coordinates.

    Returns:
        The point cloud created from the points.
    """
    return PointCloud(o3d.utility.Vector3dVector(points))


def sample_point_cloud_from_triangle_mesh(mesh_or_filename: Union[TriangleMesh, str],
                                          number_of_points: int,
                                          sample_type: Union[SampleTypes, None] = SampleTypes.UNIFORMLY,
                                          **kwargs: Any) -> PointCloud:
    """Convenience function to obtain point clouds from triangle meshes.

    Args:
        mesh_or_filename: The triangle mesh from which the point cloud is sampled.
        number_of_points: Number of points to sample from the triangle mesh.
        sample_type: How to sample the points from the triangle mesh. If `None`, returns mesh vertices.

    Returns:
        The point cloud obtained from the triangle mesh through sampling.
    """
    if isinstance(mesh_or_filename, str):
        mesh = read_triangle_mesh(filename=mesh_or_filename, **kwargs)
    elif isinstance(mesh_or_filename, TriangleMesh):
        mesh = mesh_or_filename
    else:
        raise TypeError(f"Can't read mesh of type {type(mesh_or_filename)}.")

    if sample_type is None:
        return PointCloud(o3d.utility.Vector3dVector(mesh.vertices))

    if sample_type == SampleTypes.UNIFORMLY:
        return mesh.sample_points_uniformly(number_of_points=number_of_points,
                                            use_triangle_normal=kwargs.get("use_triangle_normal", False))
    elif sample_type == SampleTypes.POISSON_DISK:
        return mesh.sample_points_poisson_disk(number_of_points=number_of_points,
                                               init_factor=kwargs.get("init_factor", 5),
                                               pcl=kwargs.get("pcl"),
                                               use_triangle_normal=kwargs.get("use_triangle_normal", False))
    else:
        raise ValueError(f"Sample type {sample_type} not supported.")


def get_camera_intrinsic_from_array(image_or_path: ImageTypes,
                                    camera_intrinsic: Union[np.ndarray, list]) -> PinholeCameraIntrinsic:
    """Constructs camera intrinsic object from image dimensions and camara intrinsic data.

    Args:
        image_or_path: An image data as produced by the camera.
        camera_intrinsic: An array or list holding the camera intrinsic parameters: fx, fy, cx, cy and s.

    Returns:
        The camera intrinsic object.
    """
    intrinsic = np.asarray(camera_intrinsic).flatten()
    assert intrinsic.size in [4, 5, 9], f"Camera intrinsic must be 4, 5 or 9 values but is {intrinsic.size}."
    if intrinsic.size == 9:
        intrinsic = intrinsic.reshape(3, 3)
    else:
        _intrinsic = intrinsic
        intrinsic = np.zeros(9).reshape(3, 3)
        intrinsic[0, 0] = _intrinsic[0]
        intrinsic[1, 1] = _intrinsic[1]
        intrinsic[0, 2] = _intrinsic[2]
        intrinsic[1, 2] = _intrinsic[3]
        if _intrinsic.size == 5:
            intrinsic[0, 1] = _intrinsic[4]
    image = eval_image_type(image_or_path=image_or_path)
    if isinstance(image, RGBDImage):
        image = image.depth
    height, width = np.asarray(image).shape[:2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    return PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)


# noinspection PyTypeChecker
def get_rgbd_image(color: ImageTypes, depth: Union[ImageTypes, None] = None, **kwargs: Any) -> RGBDImage:
    """Constructs an RGB-D image object from color and depth data.

    Args:
        color: The RGB image data.
        depth: The depth image data. If `None`, `color` needs to be RGB-D, i.e. needs to contain a depth channel.

    Returns:
        The RGB-D image object.
    """
    if depth is None and isinstance(color, RGBDImage):
        return color

    if isinstance(color, str):
        _color = np.asarray(o3d.io.read_image(color))
    elif isinstance(color, Image):
        _color = np.asarray(color)
    elif isinstance(color, np.ndarray):
        _color = color
    else:
        raise TypeError(f"`color` must have type {ImageTypes} but has type {type(color)}.")

    if isinstance(depth, str):
        _depth = np.asarray(o3d.io.read_image(depth))
    elif isinstance(depth, Image):
        _depth = np.asarray(depth)
    elif isinstance(depth, np.ndarray):
        _depth = depth
    else:
        raise TypeError(f"`depth` must have type {ImageTypes} or `None` but has type {type(depth)}.")

    return RGBDImage().create_from_color_and_depth(color=Image(_color[:, :, :3]) if _depth is None else Image(_color),
                                                   depth=Image(_color[:, :, 3]) if _depth is None else Image(_depth),
                                                   depth_scale=kwargs.get("depth_scale", 1000.0),
                                                   depth_trunc=kwargs.get("depth_trunc", 3.0),
                                                   convert_rgb_to_intensity=kwargs.get("convert_rgb_to_intensity",
                                                                                       True))


def eval_image_type(image_or_path: RGBDImageTypes, **kwargs: Any) -> Union[Image, RGBDImage]:
    """Convenience function constructing an RGB or RGB-D image based on input type.

    Args:
        image_or_path: The image data. Color and depth for RGB-D.

    Returns:
        The evaluated image object.
    """
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
        else:
            raise TypeError(f"Image type {type(image_or_path)} not supported.")

        if len(color_or_depth.shape) in [2, 3]:
            return Image(color_or_depth)
        else:
            raise ValueError(f"Input shape must be WxH, WxHx3 or WxHx4 but is {color_or_depth.shape}.")


def eval_camera_intrinsic_type(image_or_path: ImageTypes, camera_intrinsic: CameraTypes) -> PinholeCameraIntrinsic:
    """Convenience function constructing a camera intrinsic object based on input type.

    Args:
        image_or_path: The image data as produced by the camera.
        camera_intrinsic: The camera intrinsic data.

    Returns:
        The evaluated camera intrinsic object.
    """
    if isinstance(camera_intrinsic, (PinholeCameraIntrinsic, PinholeCameraIntrinsicParameters)):
        return camera_intrinsic
    elif isinstance(camera_intrinsic, (np.ndarray, list, tuple)):
        return get_camera_intrinsic_from_array(image_or_path=image_or_path, camera_intrinsic=camera_intrinsic)
    elif isinstance(camera_intrinsic, str):
        return get_camera_intrinsic_from_blenderproc_bopwriter(path_to_camera_json=camera_intrinsic)
    else:
        raise TypeError(f"Camera intrinsic type {type(camera_intrinsic)} not supported.")


def convert_depth_image_to_point_cloud(image_or_path: ImageTypes,
                                       camera_intrinsic: CameraTypes,
                                       camera_extrinsic: Union[np.ndarray, list, None] = np.eye(4),
                                       depth_scale: float = 1000.0,
                                       depth_trunc: float = 1000.0,
                                       **kwargs: Any) -> PointCloud:
    """Convenience function converting depth images to point clouds.

    Args:
        image_or_path: The depth image data.
        camera_intrinsic: The camera intrinsic data.
        camera_extrinsic: The camera extrinsic transformation matrix.
        depth_scale: The scale of the depth data. 1000.0 means it is in millimeters and will be converted to meters.
        depth_trunc: The distance at which to truncate the depth when creating the point cloud.

    Returns:
        The point cloud created from the depth image data.
    """
    image = eval_image_type(image_or_path=image_or_path, **kwargs)
    assert isinstance(image, Image), f"'image' must have type 'Image' but has type {type(image)}."
    assert len(np.asarray(image).shape) == 2, f"Depth image must have shape WxH but is {np.asarray(image).shape}."

    intrinsic = eval_camera_intrinsic_type(image_or_path=image, camera_intrinsic=camera_intrinsic)
    if camera_extrinsic is None:
        extrinsic = np.eye(4)
    else:
        extrinsic = np.asarray(camera_extrinsic).reshape(4, 4)

    return PointCloud().create_from_depth_image(depth=image,
                                                intrinsic=intrinsic,
                                                extrinsic=extrinsic,
                                                depth_scale=depth_scale,
                                                depth_trunc=depth_trunc,
                                                stride=kwargs.get("stride", 1),
                                                project_valid_depth_only=kwargs.get("project_valid_depth_only", True))


def convert_rgbd_image_to_point_cloud(rgbd_image_or_path: RGBDImageTypes,
                                      camera_intrinsic: CameraTypes,
                                      camera_extrinsic: Union[np.ndarray, list, None] = np.eye(4),
                                      depth_scale: float = 1000.0,
                                      depth_trunc: float = 1000.0,
                                      **kwargs: Any) -> PointCloud:
    """Convenience function converting RGB-D image data to point clouds.

    Args:
        rgbd_image_or_path: The color and depth image data.
        camera_intrinsic: The camera intrinsic data.
        camera_extrinsic: The camera extrinsic transformation matrix.
        depth_scale: The scale of the depth data. 1000.0 means it is in millimeters and will be converted to meters.
        depth_trunc: The distance at which to truncate the depth when creating the point cloud.

    Returns:
        The point cloud created from the RGB-D image data.
    """
    rgbd_image = eval_image_type(image_or_path=rgbd_image_or_path,
                                 depth_scale=depth_scale,
                                 depth_trunc=depth_trunc,
                                 **kwargs)
    assert isinstance(rgbd_image, RGBDImage), f"'rgbd_image must have type 'RGBDImage' but has type {type(rgbd_image)}."

    intrinsic = eval_camera_intrinsic_type(image_or_path=rgbd_image, camera_intrinsic=camera_intrinsic)
    if camera_extrinsic is None:
        extrinsic = np.eye(4)
    else:
        extrinsic = np.asarray(camera_extrinsic).reshape(4, 4)

    return PointCloud().create_from_rgbd_image(image=rgbd_image,
                                               intrinsic=intrinsic,
                                               extrinsic=extrinsic,
                                               project_valid_depth_only=kwargs.get("project_valid_depth_only", True))


def eval_transformation_data(transformation_data: TransformationTypes) -> np.ndarray:
    """Evaluates different types of transformation data to obtain a 4x4 transformation matrix.

    Args:
        transformation_data: Array or list(s) containing transformation (rotation, translation) data.

    Returns:
        A 4x4 transformation matrix or "center".
    """
    if isinstance(transformation_data, str):
        if transformation_data.lower() == "center":
            return np.asarray("center")
        data = transformation_data if os.path.exists(transformation_data) else eval(transformation_data)
    else:
        data = transformation_data

    if isinstance(data, np.ndarray):
        if np.array_equal(data, np.asarray("center")):
            return data
        if data.size == 16:
            return data.reshape(4, 4)
        elif data.size in [3, 4]:
            T = np.eye(4)
            T[:3, 3] = data.ravel()[:3]
            return T
        elif data.size == 9:
            T = np.eye(4)
            T[:3, :3] = data.reshape(3, 3)
            return T
        else:
            raise ValueError(f"Transformation data needs 3, 4, 9 or 16 values but has {data.size}.")
    elif isinstance(data, list):
        if len(data) == 2:
            if len(data[0]) == 3 and len(data[1]) >= 3:
                return get_transformation_matrix_from_xyz(rotation_xyz=data[0], translation_xyz=data[1])
            elif len(data[0]) == 4 and len(data[1]) >= 3:
                return get_transformation_matrix_from_quaternion(rotation_wxyz=data[0], translation_xyz=data[1])
            elif len(data[0]) == 9 and len(data[1]) >= 3:
                T = np.eye(4)
                T[:3, :3] = np.asarray(data[0]).reshape(3, 3)
                T[:3, 3] = np.asarray(data[1]).ravel()[:3]
                return T
            else:
                raise ValueError(f"Transformation needs 3, 4 or 9 rotation values and 3 or 4 translation values.")
        elif len(data) == 3:
            logger.debug("Ambiguous input. Could be XYZ Euler angles or XYZ translation. Interpreting as translation.")
            T = np.eye(4)
            T[:3, 3] = data
            return T
        elif len(data) == 4:
            if data[-1] == 1.0:
                T = np.eye(4)
                T[:3, 3] = data[:3]
                return T
            else:
                return get_transformation_matrix_from_quaternion(rotation_wxyz=data)
        elif len(data) == 6:
            return get_transformation_matrix_from_xyz(rotation_xyz=data[:3], translation_xyz=data[3:])
        elif len(data) == 7:
            return get_transformation_matrix_from_quaternion(rotation_wxyz=data[:4], translation_xyz=data[4:])
        elif len(data) == 9:
            T = np.eye(4)
            T[:3, :3] = np.asarray(data).reshape(3, 3)
            return T
        elif len(data) in [12, 16]:
            T = np.eye(4)
            T[:3, :3] = np.asarray(data[:9]).reshape(3, 3)
            T[:3, 3] = data[9:12]
            return T
    elif isinstance(data, str):
        return get_ground_truth_pose_from_file(path_to_ground_truth_json=data)
    else:
        raise TypeError(f"Transformation data of unsupported type {type(data)}.")


def get_transformation_matrix_from_xyz(rotation_xyz: Union[np.array, list] = np.zeros(3),
                                       translation_xyz: Union[np.array, list] = np.zeros(3)) -> np.ndarray:
    """Constructs a 4x4 homogenous transformation matrix from a XYZ translation vector and XYZ Euler angles.

    Args:
        rotation_xyz: The XYZ Euler angles in degrees.
        translation_xyz: The XYZ translation vector.

    Returns:
        The 4x4 homogenous transformation matrix.
    """
    rx, ry, rz = np.asarray(rotation_xyz).ravel()[:3]
    tx, ty, tz = np.asarray(translation_xyz).ravel()[:3]
    R = PointCloud().get_rotation_matrix_from_xyz(np.array([np.radians(rx), np.radians(ry), np.radians(rz)]))
    t = [tx, ty, tz]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def get_transformation_matrix_from_quaternion(rotation_wxyz: Union[np.array, list] = np.zeros(4),
                                              translation_xyz: Union[np.array, list] = np.zeros(3)) -> np.ndarray:
    """Constructs a 4x4 homogenous transformation matrix from a XYZ translation vector and WXYZ quaternion values.

    Args:
        rotation_wxyz: The WXYZ quaternion values.
        translation_xyz: The XYZ translation vector.

    Returns:
        The 4x4 homogenous transformation matrix.
    """
    rotation = np.asarray(rotation_wxyz).ravel()[:4]
    tx, ty, tz = np.asarray(translation_xyz).ravel()[:3]
    R = PointCloud().get_rotation_matrix_from_quaternion(rotation=rotation)
    t = [tx, ty, tz]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def get_ground_truth_pose_from_file(path_to_ground_truth_json: str) -> np.ndarray:
    """Reads ground truth from JSON file. Must contain keys starting with 'rot' (rotation) and 'tra' (translation).

    Args:
        path_to_ground_truth_json: Path to the ground truth JSON file.

    Returns:
        The ground truth pose as 4x4 transformation matrix.
    """
    with open(path_to_ground_truth_json) as f:
        ground_truth = json.load(f)

    assert any(key.startswith("rot") for key in ground_truth) and any(key.startswith("tra") for key in ground_truth), \
        f"No key starting with 'rot' and/or 'tra' found in ground truth data."
    for key, value in ground_truth.items():
        if key.startswith("rot"):
            R = value
        elif key.startswith("tra"):
            t = value

    return eval_transformation_data(transformation_data=R + t)


def get_ground_truth_pose_from_blenderproc_bopwriter(path_to_scene_gt_json: str,
                                                     scene_id: Union[int, str, List[int], List[str]] = -1,
                                                     object_id: Union[int, str, List[int], List[str]] = -1,
                                                     mm_to_m: bool = True) -> Dict[int, Dict[int, List[np.ndarray]]]:
    """Reads ground truth poses from BlenderProc BopWriter data.

    Args:
        path_to_scene_gt_json: The path to the `scene_gt.json` file.
        scene_id: The ID(s) of the rendered scene(s). Returns all scenes if set to -1.
        object_id: The ID(s) of the object(s) in the scene(s). Returns poses of all object if set to -1.
        mm_to_m: Transform BlenderProc BopWriter data in millimeter scale to meter scale.

    Returns:
        The ground truth pose of each object with `scene_id` in the scene with `scene_id`.
    """
    with open(path_to_scene_gt_json) as f:
        ground_truth = json.load(f)

    scene_id = [int(sid) for sid in scene_id] if isinstance(scene_id, list) else [int(scene_id)]
    object_id = [int(oid) for oid in object_id] if isinstance(object_id, list) else [int(object_id)]
    gt_dict = dict()
    for key, value in ground_truth.items():
        if int(key) in scene_id or scene_id[0] == -1:
            scene_dict = dict()
            for gt in value:
                if int(gt["obj_id"]) in object_id or object_id[0] == -1:
                    T = np.eye(4)
                    T[:3, :3] = np.asarray(gt["cam_R_m2c"]).reshape(3, 3)
                    T[:3, 3] = np.asarray(gt["cam_t_m2c"]) / (1000.0 if mm_to_m else 1.0)
                    if int(gt["obj_id"]) in scene_dict:
                        scene_dict[int(gt["obj_id"])].append(T)
                    else:
                        scene_dict[int(gt["obj_id"])] = [T]
            if scene_dict:
                gt_dict[int(key)] = scene_dict
    return gt_dict


def get_camera_intrinsic_from_blenderproc_bopwriter(path_to_camera_json: str,
                                                    output_path: Union[str, None] = None) -> PinholeCameraIntrinsic:
    """Constructs intrinsic camera parameter object from BlenderProc BopWriter data.

    Args:
        path_to_camera_json: The path to the camera JSON file holding the camera intrinsic parameters.
        output_path: Optional output path, where the resulting intrinsic camera parameters are stored.

    Returns:
        The camera intrinsic parameter object.
    """
    with open(path_to_camera_json) as f:
        intrinsic_data = json.load(f)

    width, height = intrinsic_data['width'], intrinsic_data['height']
    cx, cy = intrinsic_data['cx'], intrinsic_data['cy']
    fx, fy = intrinsic_data['fx'], intrinsic_data['fy']
    intrinsic = PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    if output_path is not None:
        o3d.io.write_pinhole_camera_intrinsic(filename=os.path.join(output_path, "camera_intrinsic.json"),
                                              intrinsic=intrinsic)
    return intrinsic


def get_camera_extrinsic_from_blenderproc_bopwriter(path_to_scene_camera_json: str,
                                                    scene_id: Union[int, str, List[int], List[str]] = -1,
                                                    mm_to_m: bool = True) -> List[np.ndarray]:
    """Reads camera extrinsic transform from BlenderProc Bopwriter data.

    Args:
        path_to_scene_camera_json: The path to the camera JSON file holding the extrinsic camera parameters of the
                                   selected scene.
        scene_id: The ID(s) of the rendered scene(s). Returns all scenes if set to -1.
        mm_to_m: Transform BlenderProc BopWriter data in millimeter scale to meter scale.

    Returns:
        List of 4x4 camera extrinsic transformations for each requested scene.
    """
    with open(path_to_scene_camera_json) as f:
        extrinsic_data = json.load(f)

    if int(scene_id) == -1:
        scene_id = [str(sid) for sid in extrinsic_data.keys()]
    else:
        scene_id = [str(sid) for sid in scene_id] if isinstance(scene_id, list) else [str(scene_id)]

    camera_extrinsic_list = list()
    for sid in scene_id:
        R_w2c = np.asarray(extrinsic_data[sid]['cam_R_w2c']).reshape(3, 3)
        t_w2c = np.asarray(extrinsic_data[sid]['cam_t_w2c']) / (1000.0 if mm_to_m else 1.0)
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c
        extrinsic = T_w2c
        camera_extrinsic_list.append(extrinsic)
    return camera_extrinsic_list


def get_camera_parameters_from_blenderproc_bopwriter(path_to_scene_camera_json: str,
                                                     path_to_camera_json: str,
                                                     scene_id: Union[int, str, List[int], List[str]] = -1,
                                                     output_path: Union[str, None] = None,
                                                     mm_to_m: bool = True) -> List[PinholeCameraParameters]:
    """Constructs intrinsic and extrinsic camera parameter object from BlenderProc BopWriter data.

    Args:
        path_to_scene_camera_json: The path to the camera JSON file holding the extrinsic camera parameters of the
                                   selected scene.
        path_to_camera_json: The path to the camera JSON file holding the camera intrinsic parameters.
        scene_id: The ID(s) of the rendered scene(s). Returns all scenes if set to -1.
        output_path: Optional output path, where the resulting intrinsic and extrinsic camera parameters are stored.
        mm_to_m: Transform BlenderProc BopWriter data in millimeter scale to meter scale.

    Returns:
        The camera intrinsic and extrinsic parameter object for each requested scene.
    """
    intrinsic = get_camera_intrinsic_from_blenderproc_bopwriter(path_to_camera_json=path_to_camera_json,
                                                                output_path=output_path)
    extrinsics = get_camera_extrinsic_from_blenderproc_bopwriter(scene_id=scene_id,
                                                                 path_to_scene_camera_json=path_to_scene_camera_json,
                                                                 mm_to_m=mm_to_m)
    with open(path_to_scene_camera_json) as f:
        extrinsic_data = json.load(f)

    if int(scene_id) == -1:
        scene_id = [str(sid) for sid in extrinsic_data.keys()]
    else:
        scene_id = [str(sid) for sid in scene_id] if isinstance(scene_id, list) else [str(scene_id)]

    camera_parameters_list = list()
    for scene, extrinsic in zip(scene_id, extrinsics):
        camera_parameters = PinholeCameraParameters()
        camera_parameters.intrinsic = intrinsic
        camera_parameters.extrinsic = extrinsic
        camera_parameters_list.append(camera_parameters)

        if output_path is not None:
            o3d.io.write_pinhole_camera_parameters(filename=os.path.join(output_path,
                                                                         f"camera_parameters_scene_{scene}.json"),
                                                   parameters=camera_parameters)
    return camera_parameters_list


"""
def get_from_blenderproc_bopwriter(path_to_bop_data_dir: str,
                                   object_name: str,
                                   ground_truth_pose: bool = True,
                                   camera_parameters: bool = True,
                                   scene_id: int = -1,
                                   object_id: int = -1) -> Dict[Any]:
    raise NotImplementedError
    path_to_scene_gt_json = os.path.join(path_to_bop_data_dir, object_name, "train_pbr/000000/scene_gt.json")
    return_dict = dict()
    if ground_truth_pose:
        return_dict["ground_truth_pose"] = get_ground_truth_pose_from_blenderproc_bopwriter()
    # Todo: Finish this
"""


def read_camera_parameters(filename: str) -> PinholeCameraParameters:
    """Reads pinhole camera parameters (intrinsic, extrinsic) from file written by
    `open3d.io.write_pinhole_camera_parameters`.

    Args:
        filename: Path to camera parameters file.

    Returns:
        The intrinsic and extrinsic pinhole camera parameters object.
    """
    return o3d.io.read_pinhole_camera_parameters(filename=filename)


def read_camera_intrinsic(filename: str) -> PinholeCameraIntrinsic:
    """Reads pinhole camera intrinsic parameters from file written by `open3d.io.write_pinhole_camera_intrinsic`.

    Args:
        filename: Path to camera intrinsic parameters file.

    Returns:
        The intrinsic pinhole camera parameters object.
    """
    return o3d.io.read_pinhole_camera_intrinsic(filename=filename)


def draw_geometries(geometries: List[o3d.geometry.Geometry],
                    window_name: str = "Visualizer",
                    size: Tuple[int] = (800, 600),
                    **kwargs: Any) -> None:
    """Convenience function to draw 3D geometries.

    Args:
        geometries: A list of Open3D geometry objects.
        window_name: The name of the visualization window.
        size: The width and height of the visualization window.
    """
    o3d.visualization.draw_geometries(geometries,
                                      window_name=window_name,
                                      width=size[0],
                                      height=size[1],
                                      point_show_normal=kwargs.get("point_show_normal", False),
                                      mesh_show_wireframe=kwargs.get("mesh_show_wireframe", False),
                                      mesh_show_back_face=kwargs.get("mesh_show_back_face", False))


def get_transformation_error(transformation_estimate: TransformationTypes,
                             transformation_ground_truth: TransformationTypes,
                             in_degrees: bool = True) -> Tuple[float, float]:
    """Computes the rotational and translational error between estimated and ground-truth transformation data.

    Args:
        transformation_estimate: The estimated transformation.
        transformation_ground_truth: The ground-truth transformation.
        in_degrees: Return rotational error in degrees instead of radians.

    Returns:
        Rotational and translation error between estimated and ground-truth transformation.
    """
    T_est = eval_transformation_data(transformation_data=transformation_estimate)
    T_gt = eval_transformation_data(transformation_data=transformation_ground_truth)
    error_rot = get_rotation_error(rotation_estimate=T_est[:3, :3],
                                   rotation_ground_truth=T_gt[:3, :3],
                                   in_degrees=in_degrees)
    error_trans = get_translation_error(translation_estimate=T_est[:3, 3],
                                        translation_ground_truth=T_gt[:3, 3])
    return error_rot, error_trans


def get_rotation_error(rotation_estimate: np.ndarray,
                       rotation_ground_truth: np.ndarray,
                       in_degrees: bool = True) -> float:
    """Computes the error between estimated and ground-truth rotation in degrees or radians.

    Args:
        rotation_estimate: The estimated rotation.
        rotation_ground_truth: The ground-truth rotation.
        in_degrees: Return rotational error in degrees instead of radians.

    Returns:
        Error between estimated and ground-truth rotation in degrees or radians.
    """
    assert (rotation_estimate.shape == rotation_ground_truth.shape == (3, 3)),\
        f"Rotation estimate and ground truth both need to have shape (3, 3) but are {rotation_estimate.shape} and " \
        f"{rotation_ground_truth.shape}."
    error_cos = 0.5 * (np.trace(rotation_estimate @ rotation_ground_truth.T) - 1.0)

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error_rad = math.acos(error_cos)
    if in_degrees:
        return np.rad2deg(error_rad)  # Convert [rad] to [deg].
    return error_rad


def get_translation_error(translation_estimate: np.ndarray, translation_ground_truth: np.ndarray) -> float:
    """Computes the mean-squared translational error between estimated and ground-truth translation.

    Args:
        translation_estimate: The estimated translation.
        translation_ground_truth: The ground-truth translation.

    Returns:
        Mean-squared-error between estimated and ground-truth translation.
    """
    assert (translation_estimate.size == translation_ground_truth.size == 3),\
        f"Translation estimate and ground truth need to have size 3 but have {translation_estimate.size} and " \
        f"{translation_ground_truth.size}."
    error = np.linalg.norm(translation_ground_truth - translation_estimate)
    return error
