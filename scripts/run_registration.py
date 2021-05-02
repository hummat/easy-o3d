#!/usr/bin/env python3
"""Performs point cloud registration using registration algorithms from this package."""

from easy_o3d import utils, registration, set_logger_level
import os
import numpy as np
import argparse
import configparser
import logging
import time
import glob
import tabulate
from typing import List, Union
import copy

logger = logging.getLogger(__name__)


def eval_estimation_method(estimation_method: str) -> registration.ICPTypes:
    if estimation_method.lower() in ["point", "pointtopoint", "point_to_point"]:
        return registration.ICPTypes.POINT
    elif estimation_method.lower() in ["plane", "pointtoplane", "point_to_plane"]:
        return registration.ICPTypes.PLANE
    elif estimation_method.lower() in ["color", "colored", "coloreicp", "coloredicp", "color_icp", "colored_icp"]:
        return registration.ICPTypes.COLOR
    else:
        raise ValueError(
            f"`estimation_method` must be `point_to_point`, `point_to_plane` or 'color' but is {estimation_method}.")


def eval_kernel(kernel: Union[str, None]) -> registration.KernelTypes:
    if kernel is None:
        return registration.KernelTypes.NONE
    elif isinstance(kernel, str):
        if kernel.lower() == "none":
            return registration.KernelTypes.NONE
        if kernel.lower() == "tukey":
            return registration.KernelTypes.TUKEY
        elif kernel.lower() == "cauchy":
            return registration.KernelTypes.CAUCHY
        elif kernel.lower() == "l1":
            return registration.KernelTypes.L1
        elif kernel.lower() == "l2":
            return registration.KernelTypes.L2
        elif kernel.lower == "gm":
            return registration.KernelTypes.GM
        else:
            raise ValueError(f"`kernel` must be empty or one of 'Tukey', 'Cauchy', 'L1', 'L2' or 'GM' but is {kernel}.")


def eval_checkers(checkers: str) -> List[registration.CheckerTypes]:
    checker_list = list()
    for checker in eval(checkers):
        if checker.lower() == "distance":
            checker_list.append(registration.CheckerTypes.DISTANCE)
        elif checker.lower() in ["edge", "edgelength", "edge_length"]:
            checker_list.append(registration.CheckerTypes.EDGE)
        elif checker.lower() in ["angle", "normal", "normalangle", "normal_angle"]:
            checker_list.append(registration.CheckerTypes.NORMAL)
        else:
            raise ValueError(f"`checkers` need to by in 'distance', 'edge' or 'normal' but are {eval(checkers)}.")
    return checker_list


def eval_number_of_points(number_of_points: str) -> Union[int, List[int]]:
    _number_of_points = eval(number_of_points)
    if isinstance(_number_of_points, list):
        if len(_number_of_points) == 1:
            return int(_number_of_points[0])
        return [int(points) for points in _number_of_points]
    return int(_number_of_points)


def eval_downsample(downsample: Union[str, None]) -> utils.DownsampleTypes:
    if downsample is None:
        return utils.DownsampleTypes.NONE
    elif isinstance(downsample, str):
        if downsample.lower() == "none":
            return utils.DownsampleTypes.NONE
        elif downsample.lower() == "voxel":
            return utils.DownsampleTypes.VOXEL
        elif downsample.lower() in ["uniform", "uniformly"]:
            return utils.DownsampleTypes.UNIFORM
        else:
            raise ValueError(f"`downsample` must be empty or one of 'none', 'voxel' or 'uniform' but is {downsample}.")


def eval_outlier(outlier: Union[str, None]) -> utils.OutlierTypes:
    if outlier is None:
        return utils.OutlierTypes.NONE
    elif isinstance(outlier, str):
        if outlier.lower() == "none":
            return utils.OutlierTypes.NONE
        elif outlier.lower() == "radius":
            return utils.OutlierTypes.RADIUS
        elif outlier.lower() in ["statistic", "statistical"]:
            return utils.OutlierTypes.STATISTICAL
        else:
            raise ValueError(f"`outlier` must be empty or one of 'none', 'radius' or 'statistical' but is {outlier}.")


def eval_orient_normals(orient_normals: Union[str, None]) -> utils.OrientationTypes:
    if orient_normals is None:
        return utils.OrientationTypes.NONE
    elif isinstance(orient_normals, str):
        if orient_normals.lower() == "none":
            return utils.OrientationTypes.NONE
        elif orient_normals.lower() in ["tangent", "tangentplane", "tangent_plane"]:
            return utils.OrientationTypes.TANGENT_PLANE
        elif orient_normals.lower() == "camera":
            return utils.OrientationTypes.CAMERA
        elif orient_normals.lower() == "direction":
            return utils.OrientationTypes.DIRECTION
        else:
            raise ValueError(f"`orient_normals` must be empty or one of 'none', 'tangent', 'camera' or 'direction' but"
                             f"is {orient_normals}.")


def eval_sample_type(sample_type: str) -> utils.SampleTypes:
    if sample_type.lower() in ["uniform", "uniformly", "none"]:
        return utils.SampleTypes.UNIFORMLY
    elif sample_type.lower() in ["poisson", "disk", "poissondisk", "poisson_disk"]:
        return utils.SampleTypes.POISSON_DISK
    else:
        raise ValueError(f"`sample_type` must be one of 'uniformly' or 'poisson_disk' but is {sample_type}.")


def eval_search_param(search_param: str) -> utils.SearchParamTypes:
    if search_param.lower() in ["hybrid", "none"]:
        return utils.SearchParamTypes.HYBRID
    elif search_param.lower() in ["knn", "nn", "nearestneighbor", "knearestneighbor", "nearest_neighbor",
                                  "k_nearest_neighbor"]:
        return utils.SearchParamTypes.KNN
    elif search_param.lower() == "radius":
        return utils.SearchParamTypes.RADIUS
    else:
        raise ValueError(f"`search_param` must be one of 'hybrid', 'knn' or 'radius' but is {search_param}.")


def eval_init_poses_or_ground_truth(poses: Union[str, None]) -> Union[List[np.ndarray], None]:
    if poses is None:
        return None
    try:
        poses = eval(poses)
        if isinstance(poses[0], list):
            if len(poses[0]) in [3, 4, 9]:
                return [utils.eval_transformation_data(transformation_data=poses)]
            return [utils.eval_transformation_data(transformation_data=pose) for pose in poses]
        else:
            return [utils.eval_transformation_data(transformation_data=pose) for pose in poses]
    except (NameError, SyntaxError):
        if os.path.exists(poses):
            if "scene_gt.json" in poses:
                poses = utils.get_ground_truth_pose_from_blenderproc_bopwriter(path_to_scene_gt_json=poses)
                poses = [list(pose.values()) for pose in list(poses.values())]
                poses = [pose for sublist in poses for pose in sublist]
                return [pose for sublist in poses for pose in sublist]
            return [utils.eval_transformation_data(transformation_data=poses)]
        elif poses.lower() == "none":
            return None
        return [utils.eval_transformation_data(transformation_data=pose) for pose in glob.glob(poses)]


def main():
    # Evaluate command line and config arguments
    start = time.time()
    parser = argparse.ArgumentParser(description="Performs point cloud registration.")
    parser.add_argument("-c", "--config", default="registration.ini", type=str,
                        help="Path to registration config.file.")
    parser.add_argument("--verbose", action="store_true", help="Get verbose output during execution.")
    args = parser.parse_args()

    config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    config.read(args.config)

    data = config["data"]
    algorithms = config["algorithms"]
    options = config["options"]

    # Enable verbose output
    if args.verbose or options.getboolean("verbose"):
        logger.setLevel(logging.DEBUG)
        set_logger_level(logging.DEBUG)

    # Instantiate initializer
    initializer = algorithms.get("initializer")
    params = config["initializer_params"]
    if initializer is not None:
        logger.debug(f"Loading initializer {initializer}.")
        n_times_initializer = params.getint("n_times")
        draw_initializer = params.getboolean("draw")
        overwrite_colors_initializer = params.getboolean("overwrite_colors")
        if initializer.lower() == "ransac":
            initializer = registration.RANSAC(max_iteration=params.getint("max_iteration"),
                                              confidence=params.getfloat("confidence"),
                                              max_correspondence_distance=params.getfloat("max_correspondence_distance"),
                                              with_scaling=params.getboolean("with_scaling"),
                                              ransac_n=params.getint("ransac_n"),
                                              checkers=eval_checkers(params.get("checkers")),
                                              similarity_threshold=params.getfloat("similarity_threshold"),
                                              normal_angle_threshold=params.getfloat("normal_angle_threshold"))
        elif initializer.lower() in ["fgr", "fast_global_registration", "fastglobalregistration"]:
            initializer = registration.FastGlobalRegistration(max_iteration=params.getint("max_iteration"),
                                                              max_correspondence_distance=params.getfloat("max_correspondence_distance"))
        else:
            raise ValueError(f"Only `ransac` and `fgr` supported as `initializer` but is {initializer}.")

    # Instantiate refiner
    refiner = algorithms.get("refiner")
    params = config["refiner_params"]
    if refiner is not None:
        logger.debug(f"Loading refiner {refiner}.")
        crop_target_around_source = params.getboolean("crop_target_around_source")
        crop_scale = params.getfloat("crop_scale")
        n_times_refiner = params.getint("n_times")
        multi_scale_refiner = params.getboolean("multi_scale")
        scales = eval(params.get("scales"))
        iterations = eval(params.get("iterations"))
        radius_multiplier = eval(params.get("radius_multiplier"))
        draw_refiner = params.getboolean("draw")
        overwrite_colors_refiner = params.getboolean("overwrite_colors")
        if refiner.lower() in ["icp", "iterative_closest_point", "iterativeclosestpoint"]:
            refiner = registration.IterativeClosestPoint(relative_fitness=params.getfloat("relative_fitness"),
                                                         relative_rmse=params.getfloat("relative_rmse"),
                                                         max_iteration=params.getint("max_iteration"),
                                                         max_correspondence_distance=params.getfloat(
                                                             "max_correspondence_distance"),
                                                         estimation_method=eval_estimation_method(
                                                             params.get("estimation_method")),
                                                         with_scaling=params.getboolean("with_scaling"),
                                                         kernel=eval_kernel(params.get("kernel")),
                                                         kernel_noise_std=params.getfloat("kernel_noise_std"))
        else:
            raise ValueError(f"Only `icp` supported as `refiner` but is {refiner}.")
    if initializer is None and refiner is None:
        raise ValueError(f"Either `initializer` or `refiner` need to be specified.")

    # Evaluate source and target data config
    logger.debug("Evaluating data.")
    source_data = data.get("source_files")
    target_data = data.get("target_files")
    try:
        source_data = eval(source_data)
    except NameError:
        source_data = [source_data] if os.path.exists(source_data) else sorted(glob.glob(source_data))
    try:
        target_data = eval(data.get("target_files"))
    except NameError:
        target_data = [target_data] if os.path.exists(target_data) else sorted(glob.glob(target_data))

    # Load source and target data
    logger.debug("Loading source data.")
    params = config["source_params"]
    number_of_points = eval_number_of_points(params.get("number_of_points"))
    camera_intrinsic = params.get("camera_intrinsic")
    camera_intrinsic = None if camera_intrinsic == "none" else eval(camera_intrinsic)
    sample_type = eval_sample_type(params.get("sample_type"))
    source_list = utils.eval_data_parallel(data=source_data,
                                           number_of_points=number_of_points,
                                           camera_intrinsic=camera_intrinsic,
                                           sample_type=sample_type)

    logger.debug("Loading target data.")
    params = config["target_params"]
    number_of_points = eval_number_of_points(params.get("number_of_points"))
    camera_intrinsic = params.get("camera_intrinsic")
    camera_intrinsic = None if camera_intrinsic == "none" else eval(camera_intrinsic)
    sample_type = eval_sample_type(params.get("sample_type"))
    target_list = utils.eval_data_parallel(data=target_data,
                                           number_of_points=number_of_points,
                                           camera_intrinsic=camera_intrinsic,
                                           sample_type=sample_type)

    # Process source and target data
    logger.debug("Processing source data.")
    params = config["source_processing"]
    source_list = utils.process_point_cloud_parallel(point_cloud_list=source_list,
                                                     downsample=eval_downsample(params.get("downsample")),
                                                     downsample_factor=params.getfloat("downsample_factor"),
                                                     remove_outlier=eval_outlier(params.get("remove_outlier")),
                                                     outlier_std_ratio=params.getfloat("outlier_std_ratio"),
                                                     scale=params.getfloat("scale"),
                                                     estimate_normals=params.getboolean("estimate_normals"),
                                                     recalculate_normals=params.getboolean("recalculate_normals"),
                                                     normalize_normals=params.getboolean("normalize_normals"),
                                                     orient_normals=eval_orient_normals(params.get("orient_normals")),
                                                     search_param=eval_search_param(params.get("search_param")),
                                                     search_param_knn=params.getint("search_param_knn"),
                                                     search_param_radius=params.getfloat("search_param_radius"),
                                                     draw=params.getboolean("draw"))

    logger.debug("Processing target data.")
    params = config["target_processing"]
    target_list = utils.process_point_cloud_parallel(point_cloud_list=target_list,
                                                     downsample=eval_downsample(params.get("downsample")),
                                                     downsample_factor=params.getfloat("downsample_factor"),
                                                     remove_outlier=eval_outlier(params.get("remove_outlier")),
                                                     outlier_std_ratio=params.getfloat("outlier_std_ratio"),
                                                     scale=params.getfloat("scale"),
                                                     estimate_normals=params.getboolean("estimate_normals"),
                                                     recalculate_normals=params.getboolean("recalculate_normals"),
                                                     normalize_normals=params.getboolean("normalize_normals"),
                                                     orient_normals=eval_orient_normals(params.get("orient_normals")),
                                                     search_param=eval_search_param(params.get("search_param")),
                                                     search_param_knn=params.getint("search_param_knn"),
                                                     search_param_radius=params.getfloat("search_param_radius"),
                                                     draw=params.getboolean("draw"))

    # Run registration algorithms
    results = list()
    if initializer is not None:
        logger.debug(f"Running initializer {initializer._name}.")
        params = config["feature_processing"]
        results = initializer.run_many(source_list=source_list,
                                       target_list=target_list,
                                       init_list=eval_init_poses_or_ground_truth(data.get("poses")),
                                       one_vs_one=algorithms.getboolean("one_vs_one"),
                                       n_times=n_times_initializer,
                                       draw=draw_initializer,
                                       overwrite_colors=overwrite_colors_initializer,
                                       search_param=eval_search_param(params.get("search_param")),
                                       search_param_knn=params.getint("search_param_knn"),
                                       search_param_radius=params.getfloat("search_param_radius"))
        init_list = [result.transformation for result in results]
    else:
        init_list = eval_init_poses_or_ground_truth(data.get("poses"))
    if refiner is not None:
        logger.debug(f"Running refiner {refiner._name}.")
        results = refiner.run_many(source_list=source_list,
                                   target_list=target_list,
                                   init_list=init_list,
                                   one_vs_one=algorithms.getboolean("one_vs_one"),
                                   n_times=n_times_refiner,
                                   multi_scale=multi_scale_refiner,
                                   source_scales=scales,
                                   iterations=iterations,
                                   radius_multiplier=radius_multiplier,
                                   crop_target_around_source=crop_target_around_source,
                                   crop_scale=crop_scale,
                                   draw=draw_refiner,
                                   overwrite_colors=overwrite_colors_refiner)
    logger.info(f"Execution took {time.time() - start} seconds.")

    ground_truth = eval_init_poses_or_ground_truth(data.get("ground_truth"))
    names = list()
    errors = list()
    if algorithms.getboolean("one_vs_one"):
        assert len(results) == len(source_list) == len(target_list)
        if ground_truth is not None:
            assert len(ground_truth) == len(results)
        for i in range(len(results)):
            names.append(f"s{i} - t{i}")
            if ground_truth is not None:
                errors.append(np.linalg.norm(ground_truth[i] - results[i].transformation))
            else:
                errors.append("TBD")
    else:
        assert len(results) == len(source_list) * len(target_list)
        if ground_truth is not None:
            assert len(ground_truth) == len(results)
        for i in range(len(target_list)):
            for j in range(len(source_list)):
                names.append(f"s{j} - t{i}")
                if ground_truth is not None:
                    errors.append(np.linalg.norm(ground_truth.pop() - copy.deepcopy(results).pop().transformation))
                else:
                    errors.append("TBD")

    table = tabulate.tabulate([(name,
                                result.fitness,
                                result.inlier_rmse,
                                len(result.correspondence_set),
                                error) for name, result, error in zip(names, results, errors)],
                              headers=["source vs. target", "fitness", "inlier rmse", "#corresp.", "error"])
    print(table)


if __name__ == "__main__":
    main()
