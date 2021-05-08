"""An easy to use wrapper around some of Open3D's registration functionality.

Files:
    __init__.py: This file.
    registration.py: Point cloud registration functionality.
    interfaces.py: Interfaces and base classes.
    utils.py: Utility functions used throughout the project.

Classes:
    registration.IterativeClosestPoint: The Iterative Closest Point (ICP) algorithm.
    registration.FastGlobalRegistration: The Fast Global Registration (FGR) algorithm.
    registration.RANSAC: The RANSAC algorithm.
    registration.ICPTypes: Supported ICP registration types.
    registration.CheckerTypes: Supported RANSAC correspondence checker types.
    registration.KernelTypes: Supported ICP Point-To-Plane robust kernel types.
    interfaces.RegistrationInterface: Interface for all registration classes.
    utils.ValueTypes: Default value flags to increase readability.
    utils.SampleTypes: Supported types of point cloud sampling from meshes.
    utils.DownsampleTypes: Supported point cloud downsampling types.
    utils.SearchParamTypes: Supported normal and FPFH feature computation search parameter types.
    utils.OrientationTypes: Supported normal orientation types.

Functions:
    get_logger: Returns the package-wide logger
    set_logger_level: Sets the package-wide logger level.
    utils.eval_data: Convenience function that automatically determines the data type and loads the data accordingly.
    utils.process_point_cloud: Utility function to apply various processing steps on point cloud data.
    utils.read_point_cloud: Reads point cloud data from file.
    utils.read_triangle_mesh: Reads triangle mesh data from file.
    utils.read_triangle_mesh_from_triangles_and_vertices: Convenience function to obtain triangle meshes from triangles
                                                          and vertices.
    utils.get_point_cloud_from_points: Convenience function to obtain point clouds from points.
    utils.sample_point_cloud_from_triangle_mesh: Convenience function to obtain point clouds from triangle meshes.
    utils.get_camera_intrinsic_from_array: Constructs camera intrinsic object from image dimensions and camara intrinsic
                                           data.
    utils.get_rgbd_image: Constructs an RGB-D image from a color and a depth image.
    utils.eval_image_type: Convenience function constructing an RGB or RGB-D image based on input type.
    utils.eval_camera_intrinsic_type: Convenience function constructing a camera intrinsic object based on input type.
    utils.convert_depth_image_to_point_cloud: Convenience function converting depth images to point clouds.
    utils.convert_rgbd_image_to_point_cloud: Convenience function converting RGB-D image data to point clouds.
    utils.get_transformation_matrix_from_xyz: Constructs a 4x4 homogenous transformation matrix from a XYZ translation
                                              vector and XYZ Euler angles
    utils.get_transformation_matrix_from_quaternion: Constructs a 4x4 homogenous transformation matrix from a XYZ
                                                     translation vector and WXYZ quaternion values.
    utils.get_camera_parameters_from_blenderproc_bopwriter: Constructs intrinsic and extrinsic camera parameter object
                                                            from BlenderProc BopWriter data.
    utils.draw_geometries: Convenience function to draw 3D geometries.
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
