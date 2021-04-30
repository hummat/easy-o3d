"""Utility functions used throughout the project.

Classes:
    SampleTypes: Supported types of point cloud sampling from meshes.
    DownsampleTypes: Supported point cloud downsampling types.
    SearchParamTypes: Supported normal and FPFH feature computation search parameter types.
    OrientationTypes: Supported normal orientation types.

Functions:
    eval_data: Convenience function that automatically determines the data type and loads the data accordingly.
    process_point_cloud: Utility function to apply various processing steps on point cloud data.
    read_point_cloud: Reads point cloud data from file.
    read_triangle_mesh: Reads triangle mesh data from file.
    read_triangle_mesh_from_triangles_and_vertices: Convenience function to obtain triangle meshes from triangles
                                                    and vertices.
    get_point_cloud_from_points: Convenience function to obtain point clouds from points.
    sample_point_cloud_from_triangle_mesh: Convenience function to obtain point clouds from triangle meshes.
    get_camera_intrinsic_from_array: Constructs camera intrinsic object from image dimensions and camara intrinsic data.
    get_rgbd_image: Constructs an RGB-D image from a color and a depth image.
    eval_image_type: Convenience function constructing an RGB or RGB-D image based on input type.
    eval_camera_intrinsic_type: Convenience function constructing a camera intrinsic object based on input type.
    convert_depth_image_to_point_cloud: Convenience function converting depth images to point clouds.
    convert_rgbd_image_to_point_cloud: Convenience function converting RGB-D image data to point clouds.
    get_transformation_matrix_from_xyz: Constructs a 4x4 homogenous transformation matrix from a XYZ translation vector
                                        and XYZ Euler angles
    get_transformation_matrix_from_quaternion: Constructs a 4x4 homogenous transformation matrix from a XYZ translation
                                               vector and WXYZ quaternion values.
    get_camera_parameters_from_blenderproc_bopwriter: Constructs intrinsic and extrinsic camera parameter object from
                                                      BlenderProc BopWriter data.
    draw_geometries: Convenience function to draw 3D geometries.
"""

import copy
import json
import logging
import os
import time
from enum import Flag, auto
from typing import Any, List, Union, Tuple

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

InputTypes = Union[str, np.ndarray, PointCloud, TriangleMesh, Image, RGBDImage]
ImageTypes = Union[Image, RGBDImage, np.ndarray, str]
RGBDImageTypes = Union[ImageTypes, List[ImageTypes]]
CameraTypes = Union[np.ndarray, PinholeCameraIntrinsic, PinholeCameraIntrinsicParameters]

logger = logging.getLogger(__name__)


class SampleTypes(Flag):
    """Supported types of point cloud sampling from meshes."""
    UNIFORMLY = auto()
    POISSON_DISK = auto()


class DownsampleTypes(Flag):
    """Supported point cloud downsampling types."""
    NONE = auto()
    VOXEL = auto()
    RANDOM = auto()
    UNIFORM = auto()


class OutlierTypes(Flag):
    """Supported outlier removal types."""
    NONE = auto()
    STATISTICAL = auto()
    RADIUS = auto()


class SearchParamTypes(Flag):
    """Supported normal and FPFH feature computation search parameter types."""
    KNN = auto()
    RADIUS = auto()
    HYBRID = auto()


class OrientationTypes(Flag):
    """Supported normal orientation types."""
    NONE = auto()
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
    elif isinstance(data, Image) and camera_intrinsic is not None:
        logger.debug("Trying to convert depth image to point cloud.")
        return convert_depth_image_to_point_cloud(image_or_path=data,
                                                  camera_intrinsic=camera_intrinsic,
                                                  **kwargs)
    elif isinstance(data, (RGBDImage, list)) and camera_intrinsic is not None:
        logger.debug("Trying to convert RGB-D image to point cloud.")
        return convert_rgbd_image_to_point_cloud(rgbd_image_or_path=data,
                                                 camera_intrinsic=camera_intrinsic,
                                                 **kwargs)
    elif isinstance(data, (str, TriangleMesh)) and number_of_points is not None:
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


def process_point_cloud(point_cloud: PointCloud,
                        downsample: DownsampleTypes = DownsampleTypes.NONE,
                        downsample_factor: Union[float, int] = 1,
                        remove_outlier: OutlierTypes = OutlierTypes.NONE,
                        outlier_std_ratio: float = 1.0,
                        transform: [np.ndarray, list, None] = None,
                        scale: float = 1.0,
                        estimate_normals: bool = False,
                        recalculate_normals: bool = False,
                        fast_normal_computation: bool = True,
                        normalize_normals: bool = False,
                        orient_normals: OrientationTypes = OrientationTypes.NONE,
                        compute_feature: bool = False,
                        search_param: SearchParamTypes = SearchParamTypes.HYBRID,
                        search_param_knn: int = 30,
                        search_param_radius: float = 0.02,  # 2cm
                        camera_location_or_direction: [np.ndarray, list] = np.zeros(3),
                        draw: bool = False) -> Union[PointCloud, Tuple[PointCloud, Feature]]:
    """Utility function to apply various processing steps on point cloud data.

    Processing steps are applied in order implied by the functions argument order:
    1. `downsample`
    2. `remove outlier`
    3. `transform`
    4. `scale`
    5. `estimate normals`
    6. `normalize normals`
    7. `orient normals`
    8. `compute feature`

    To estimate normals and compute feature with individual search parameters, rerun the function for each step.

    Args:
        point_cloud: The point cloud to be processed.
        downsample: Reduce point cloud density by dropping points uniformly or in voxel grid fashion.
        downsample_factor: The amount of downsampling. Factor for `DownsampleType.UNIFORM`, voxel size for
                           `DownsampleType.VOXEL`.
        remove_outlier: Remove outlier vertices based on radius density or variance.
        outlier_std_ratio: Standard deviation for statistical outlier removal. Smaller removes more vertices.
        transform: Homogeneous transformation. Also accepts translation vector or rotation matrix.
        scale: Scales the point cloud.
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
    if downsample != DownsampleTypes.NONE:
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

    if remove_outlier != OutlierTypes.NONE:
        num_points = len(np.asarray(_point_cloud.points))
        if remove_outlier == OutlierTypes.STATISTICAL:
            _point_cloud, _ = _point_cloud.remove_statistical_outlier(nb_neighbors=search_param_knn,
                                                                      std_ratio=outlier_std_ratio)
        elif remove_outlier == OutlierTypes.RADIUS:
            _point_cloud, _ = _point_cloud.remove_radius_outlier(nb_points=search_param_knn, radius=search_param_radius)
        else:
            raise ValueError(f"`remove_outlier` needs to be one of `OutlierTypes` but is {type(remove_outlier)}.")
        logger.debug(f"Removed {num_points - len(np.asarray(_point_cloud.points))} outliers.")

    if transform is not None:
        _transform = np.asarray(transform)
        if _transform.size in [3, 4]:
            _point_cloud.translate(translation=_transform.ravel()[:3], relative=True)
        elif _transform.size == 9:
            # noinspection PyArgumentList
            _point_cloud.rotate(R=_transform.reshape(3, 3), center=_point_cloud.get_center())
        elif _transform.size == 16:
            _point_cloud.transform(_transform.reshape(4, 4))
        else:
            raise ValueError("`transform` needs to be a valid translation, rotation or transformation in natural or"
                             "homogeneous coordinates, i.e. of size 3, 4, 9 or 16.")

    if scale != 1.0:
        logger.debug(f"Scaling point cloud with factor {scale}.")
        logger.debug("Using custom scaling code as PointCloud.scale doesn't seem to work.")
        _point_cloud.points = o3d.utility.Vector3dVector(np.asarray(_point_cloud.points) * scale)
        # FIXME: Open3D factory `scale` function doesn't do anything.
        # _point_cloud = _point_cloud.scale(scale=scale, center=_point_cloud.get_center())

    if estimate_normals or compute_feature:
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

    if orient_normals != OrientationTypes.NONE:
        assert _point_cloud.has_normals(), "Point cloud doesn't have normals which could be oriented."
        logger.debug(f"Orienting normals towards {orient_normals}.")
        if orient_normals == OrientationTypes.TANGENT_PLANE:
            _point_cloud.orient_normals_consistent_tangent_plane(k=search_param_knn)
        elif orient_normals == OrientationTypes.CAMERA:
            _point_cloud.orient_normals_towards_camera_location(camera_location=camera_location_or_direction)
        elif orient_normals == OrientationTypes.DIRECTION:
            _point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.asarray(camera_location_or_direction))
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


def read_point_cloud(filename: str, **kwargs: Any) -> PointCloud:
    """Reads point cloud data from file.

    Args:
        filename: The path to the triangle mesh file.

    Returns:
        The point cloud data read from file.
    """
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
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def sample_point_cloud_from_triangle_mesh(mesh_or_filename: Union[TriangleMesh, str],
                                          number_of_points: int,
                                          sample_type: SampleTypes = SampleTypes.UNIFORMLY,
                                          **kwargs: Any) -> PointCloud:
    """Convenience function to obtain point clouds from triangle meshes.

    Args:
        mesh_or_filename: The triangle mesh from which the point cloud is sampled.
        number_of_points: Number of points to sample from the triangle mesh.
        sample_type: How to sample the points from the triangle mesh.

    Returns:
        The point cloud obtained from the triangle mesh through sampling.
    """
    if isinstance(mesh_or_filename, str):
        mesh = read_triangle_mesh(filename=mesh_or_filename, **kwargs)
    elif isinstance(mesh_or_filename, TriangleMesh):
        mesh = mesh_or_filename
    else:
        raise TypeError(f"Can't read mesh of type {type(mesh_or_filename)}.")

    if sample_type == SampleTypes.UNIFORMLY:
        return mesh.sample_points_uniformly(number_of_points=number_of_points,
                                            use_triangle_normal=kwargs.get("use_triangle_normal", False),
                                            seed=kwargs.get("seed", -1))
    elif sample_type == SampleTypes.POISSON_DISK:
        return mesh.sample_points_poisson_disk(number_of_points=number_of_points,
                                               init_factor=kwargs.get("init_factor", 5),
                                               pcl=kwargs.get("pcl"),
                                               use_triangle_normal=kwargs.get("use_triangle_normal", False),
                                               seed=kwargs.get("seed", -1))
    else:
        raise ValueError(f"Sample style {sample_type} not supported.")


def get_camera_intrinsic_from_array(image_or_path: ImageTypes,
                                    camera_intrinsic: Union[np.ndarray, list]) -> PinholeCameraIntrinsic:
    """Constructs camera intrinsic object from image dimensions and camara intrinsic data.

    Args:
        image_or_path: An image data as produced by the camera.
        camera_intrinsic: An array holding the camera intrinsic parameters.

    Returns:
        The camera intrinsic object.
    """
    assert camera_intrinsic.size == 9, f"Camera intrinsic must be 9 values but is {camera_intrinsic.size}."
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

    if _depth is None:
        assert len(_color.shape) == 4, "Color image must have depth channel if depth is not provided."
        _color = Image(_color[:, :, :3])
        _depth = Image(_color[:, :, 3])
        return RGBDImage().create_from_color_and_depth(color=_color,
                                                       depth=_depth,
                                                       depth_scale=kwargs.get("depth_scale", 1000.0),
                                                       depth_trunc=kwargs.get("depth_trunc", 3.0),
                                                       convert_rgb_to_intensity=kwargs.get("convert_rgb_to_intensity",
                                                                                           True))
    else:
        return RGBDImage().create_from_color_and_depth(color=Image(_color),
                                                       depth=Image(_depth),
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

        if len(color_or_depth.shape) == 2:
            return Image(image_or_path)
        elif len(color_or_depth.shape) == 3:
            if color_or_depth.shape[2] == 3:
                return Image(image_or_path)
            if color_or_depth.shape[2] == 4:
                return get_rgbd_image(color=image_or_path, **kwargs)
        else:
            raise ValueError(f"Input shape must be WxH, WxHx3 or WxHx4 but is {color_or_depth.shape}.")


def eval_camera_intrinsic_type(image_or_path: ImageTypes, camera_intrinsic: CameraTypes):
    """Convenience function constructing a camera intrinsic object based on input type.

    Args:
        image_or_path: The image data as produced by the camera.
        camera_intrinsic: The camera intrinsic data.

    Returns:
        The evaluated camera intrinsic object.
    """
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
    """Convenience function converting depth images to point clouds.

    Args:
        image_or_path: The depth image data.
        camera_intrinsic: The camera intrinsic data.
        camera_extrinsic: The camera extrinsic transformation matrix.
        depth_scale: The scale of the depth data.
        depth_trunc: The distance at which to truncate the depth when creating the point cloud.

    Returns:
        The point cloud created from the depth image data.
    """
    image = eval_image_type(image_or_path=image_or_path, **kwargs)
    assert isinstance(image, Image)
    assert len(np.asarray(image).shape) == 2, f"Depth image must have shape WxH but is {image.shape}."

    intrinsic = eval_camera_intrinsic_type(image_or_path=image, camera_intrinsic=camera_intrinsic)
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
                                      camera_extrinsic: Union[np.ndarray, list] = np.eye(4),
                                      depth_scale: float = 1000.0,
                                      depth_trunc: float = 1000.0,
                                      **kwargs: Any) -> PointCloud:
    """Convenience function converting RGB-D image data to point clouds.

    Args:
        rgbd_image_or_path: The color and depth image data.
        camera_intrinsic: The camera intrinsic data.
        camera_extrinsic: The camera extrinsic transformation matrix.
        depth_scale: The scale of the depth data.
        depth_trunc: The distance at which to truncate the depth when creating the point cloud.

    Returns:
        The point cloud created from the RGB-D image data.
    """
    rgbd_image = eval_image_type(image_or_path=rgbd_image_or_path,
                                 depth_scale=depth_scale,
                                 depth_trunc=depth_trunc,
                                 **kwargs)
    assert isinstance(rgbd_image, RGBDImage)

    intrinsic = eval_camera_intrinsic_type(image_or_path=rgbd_image, camera_intrinsic=camera_intrinsic)
    extrinsic = np.asarray(camera_extrinsic).reshape(4, 4)

    return PointCloud().create_from_rgbd_image(image=rgbd_image,
                                               intrinsic=intrinsic,
                                               extrinsic=extrinsic,
                                               project_valid_depth_only=kwargs.get("project_valid_depth_only", True))


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


def get_camera_parameters_from_blenderproc_bopwriter(scene_id: Union[int, str],
                                                     path_to_scene_camera_json: str,
                                                     path_to_camera_json: str,
                                                     output_path: Union[str, None] = None) -> PinholeCameraParameters:
    """Constructs intrinsic and extrinsic camera parameter object from BlenderProc BopWriter data.

    Args:
        scene_id: The ID of the rendered scene.
        path_to_scene_camera_json: The path to the camera JSON file holding the extrinsic camera parameters of the
                                   selected scene.
        path_to_camera_json: The path to the camera JSON file holding the camera intrinsic parameters.
        output_path: Optional output path, where the resulting intrinsic and extrinsic camera parameters are stored.

    Returns:
        The camera intrinsic and extrinsic parameter object.
    """
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
    intrinsic = PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    # Set camera parameters
    camera_parameters = PinholeCameraParameters()
    camera_parameters.intrinsic = intrinsic
    camera_parameters.extrinsic = extrinsic

    # Maybe write to file
    if output_path is not None:
        o3d.io.write_pinhole_camera_parameters(os.path.join(output_path, "camera_parameters.json"), camera_parameters)

    return camera_parameters


def draw_geometries(geometries: list,
                    window_name: str = "Visualizer",
                    size: Tuple[int] = (800, 600),
                    **kwargs: Any) -> None:
    """Convenience function to draw 3D geometries.

    Args:
        geometries: A list of 3D geometry objects.
        window_name: The name of the visualization window.
        size: The size of the visualization window.
    """
    o3d.visualization.draw_geometries(geometries,
                                      window_name=window_name,
                                      width=size[0],
                                      height=size[1],
                                      point_show_normal=kwargs.get("point_show_normal", False),
                                      mesh_show_wireframe=kwargs.get("mesh_show_wireframe", False),
                                      mesh_show_back_face=kwargs.get("mesh_show_back_face", False))
