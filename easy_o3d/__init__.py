"""An easy to use wrapper around some of Open3D's registration functionality.

Files:
    __init__.py: This file.
    registration.py: Point cloud registration functionality.
    interfaces.py: Intefaces and base classes.
    utils.py: Utility functions used throughout the project.

Classes:
    registration.IterativeClosestPoint: ICP functionality.
    registration.FastGlobalRegistration: FGR functionality.
    registration.EstimationMethodTypes: Estimation method style flags.
    interfaces.RegistrationInterface: Interface for all registration classes.
    utils.ValueTypes: Flags for default values.
    utils.SampleTypes: Sample style flags.
    utils.DownsampleTypes: Downsample style flags.
    utils.SearchParamTypes: KNN and FPFH feature computation style flags.
    utils.OrientationTypes: Normal orientation style flags.

Functions:
    get_logger: Returns the package-wide logger
    set_logger_level: Sets the package-wide logger level.
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

import logging

logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    """Returns the package-wide logger.

    Returns:
        logging.Logger: The package-wide logger.
    """
    return logger


def set_logger_level(level: int) -> None:
    """Sets the package-wide logger level.

    Args:
        level (int): The logger level.
    """
    logger.setLevel(level=level)
