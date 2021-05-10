#!/usr/bin/env python3
"""Performs point cloud registration using registration algorithms from this package."""
import sys

from easy_o3d import utils, registration, set_logger_level
import os
import numpy as np
import argparse
import configparser
import logging
import time
import glob
import tabulate
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)


def eval_config(config: configparser.ConfigParser) -> Dict[str, Dict[str, Any]]:
    """Evaluates data types of a ConfigParser object.

    Args:
        config: A ConfigParser object.

    Returns:
        A dict of dicts with sections and options identical to 'config' but with evaluated values.
    """
    config_dict = dict()
    for section in config.sections():
        config_dict[section] = dict()
        for option, values in config.items(section):
            try:
                values = eval(values)
                if isinstance(values, list):
                    if section.lower() in ["source_params", "target_params"]:
                        if option.lower() == "sample_type":
                            _values = list()
                            for value in values:
                                if value is None or value.lower() == "none":
                                    _values.append(None)
                                elif "uniform" in value.lower():
                                    _values.append(utils.SampleTypes.UNIFORMLY)
                                elif "disk" in value.lower():
                                    _values.append(utils.SampleTypes.POISSON_DISK)
                            values = _values
                    elif section.lower() in ["feature_processing", "source_processing", "target_processing"]:
                        if option.lower() == "search_param":
                            _values = list()
                            for value in values:
                                if value is None or value.lower() == "none":
                                    _values.append(None)
                                elif "hybrid" in value.lower():
                                    _values.append(utils.SearchParamTypes.HYBRID)
                                elif "knn" in value.lower():
                                    _values.append(utils.SearchParamTypes.KNN)
                                elif "radius" in value.lower():
                                    _values.append(utils.SearchParamTypes.RADIUS)
                            values = _values
                        elif option.lower() == "downsample":
                            _values = list()
                            for value in values:
                                if value is None or value.lower() == "none":
                                    _values.append(None)
                                elif "uniform" in value.lower():
                                    _values.append(utils.DownsampleTypes.UNIFORM)
                                elif "voxel" in value.lower():
                                    _values.append(utils.DownsampleTypes.VOXEL)
                            values = _values
                        elif option.lower() == "remove_outlier":
                            _values = list()
                            for value in values:
                                if value is None or value.lower() == "none":
                                    _values.append(None)
                                elif "statistic" in value.lower():
                                    _values.append(utils.OutlierTypes.STATISTICAL)
                                elif "radius" in value.lower():
                                    _values.append(utils.OutlierTypes.RADIUS)
                            values = _values
                        if option.lower() == "orient_normals":
                            _values = list()
                            for value in values:
                                if value is None or value.lower() == "none":
                                    _values.append(None)
                                elif "tangent" in value.lower():
                                    _values.append(utils.OrientationTypes.TANGENT_PLANE)
                                elif "camera" in value.lower():
                                    _values.append(utils.OrientationTypes.CAMERA)
                                elif "direction" in value.lower():
                                    _values.append(utils.OrientationTypes.DIRECTION)
                            values = _values
                    elif section.lower() == "initializer_params":
                        if option.lower() == "checkers":
                            _values = list()
                            for value in values:
                                if value.lower() == "distance":
                                    _values.append(registration.CheckerTypes.DISTANCE)
                                elif value.lower() == "edge":
                                    _values.append(registration.CheckerTypes.EDGE)
                                elif value.lower() == "normal":
                                    _values.append(registration.CheckerTypes.NORMAL)
            except (NameError, SyntaxError):
                if values.lower() == "none":
                    values = None
                elif section.lower() in ["data", "source_params", "target_params"]:
                    if option.lower() in ["source_files", "target_files"]:
                        _values = [values] if os.path.exists(values) else sorted(glob.glob(values))
                        if len(_values) == 0:
                            raise FileNotFoundError(f"No files found at {values}.")
                        values = _values
                    elif option.lower() in ["ground_truth", "init_poses", "camera_intrinsic", "camera_extrinsic"]:
                        if os.path.exists(values):
                            if "scene_gt.json" in values.lower():
                                poses = utils.get_ground_truth_pose_from_blenderproc_bopwriter(values)
                                poses = [list(pose.values()) for pose in list(poses.values())]
                                poses = [pose for sublist in poses for pose in sublist]
                                values = [pose for sublist in poses for pose in sublist]
                            elif "scene_camera.json" in values.lower():
                                values = utils.get_camera_extrinsic_from_blenderproc_bopwriter(values)
                        elif values.lower() != "center":
                            values = sorted(glob.glob(values))
                    elif option.lower() == "sample_type":
                        values = utils.SampleTypes.UNIFORMLY if "uniform" in values.lower() else utils.SampleTypes.POISSON_DISK
                elif section.lower() in ["feature_processing", "source_processing", "target_processing"]:
                    if option.lower() == "search_param":
                        if values.lower() == "hybrid":
                            values = utils.SearchParamTypes.HYBRID
                        elif values.lower() == "knn":
                            values = utils.SearchParamTypes.KNN
                        elif values.lower() == "radius":
                            values = utils.SearchParamTypes.RADIUS
                    elif option.lower() == "downsample":
                        values = utils.DownsampleTypes.UNIFORM if "uniform" in values else utils.DownsampleTypes.VOXEL
                    elif option.lower() == "remove_outlier":
                        values = utils.OutlierTypes.STATISTICAL if "statistic" in values else utils.OutlierTypes.RADIUS
                    elif option.lower() == "orient_normals":
                        if values.lower() == "tangent":
                            values = utils.OrientationTypes.TANGENT_PLANE
                        elif values.lower() == "camera":
                            values = utils.OrientationTypes.CAMERA
                        elif values.lower() == "direction":
                            values = utils.OrientationTypes.DIRECTION
                elif section.lower() == "initializer_params":
                    if option.lower() == "checkers":
                        if values.lower() == "distance":
                            values = [registration.CheckerTypes.DISTANCE]
                        elif values.lower() == "edge":
                            values = [registration.CheckerTypes.EDGE]
                        elif values.lower() == "normal":
                            values = [registration.CheckerTypes.NORMAL]
                elif section.lower() == "refiner_params":
                    if option.lower() == "estimation_method":
                        if values.lower() in ["point", "point_to_point"]:
                            values = registration.ICPTypes.POINT
                        elif values.lower() in ["plane", "point_to_plane"]:
                            values = registration.ICPTypes.PLANE
                        elif values.lower() in ["color", "colored"]:
                            values = registration.ICPTypes.COLOR
                    elif option.lower() == "kernel":
                        if values.lower() == "tukey":
                            values = registration.KernelTypes.TUKEY
                        elif values.lower() == "cauchy":
                            values = registration.KernelTypes.CAUCHY
                        elif values.lower() == "l1":
                            values = registration.KernelTypes.L1
                        elif values.lower() == "l2":
                            values = registration.KernelTypes.L2
                        elif values.lower() == "gm":
                            values = registration.KernelTypes.GM
            config_dict[section][option] = values
    return config_dict


def print_config_dict(config_dict: Dict[str, Any], pretty: bool = True) -> None:
    """Pretty-prints a config dict created by 'eval_config'.

    Args:
        config_dict: A config dict created by 'eval_config'.
        pretty: Pretty-print dict keys.
    """
    config_list = list()
    for section in config_dict.keys():
        config_list.append(("", ""))
        config_list.append((section.upper().replace('_', ' ') if pretty else section, ""))
        config_list.append(('-' * len(section), ""))
        for key, value in config_dict[section].items():
            value = str(value)
            config_list.append((key.capitalize().replace('_', ' ') if pretty else key,
                                value.capitalize() if value.lower() in ["true", "false", "none"] and pretty else value))
    print(tabulate.tabulate(config_list))


def run(config: Union[configparser.ConfigParser, None] = None) -> Dict[str, Any]:
    # Evaluate command line arguments
    start = time.time()
    parser = argparse.ArgumentParser(description="Performs point cloud registration.")
    parser.add_argument("-c", "--config",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "registration.ini"), type=str,
                        help="Path to registration config.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Get verbose output during execution.")
    parser.add_argument("-d", "--draw", action="store_true", help="Visualize registration results.")
    args = parser.parse_args()

    # Read config from argument or file
    if config is None:
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(args.config)

    # Evaluate config
    config_dict = eval_config(config)
    data = config_dict["data"]
    source_params = config_dict["source_params"]
    target_params = config_dict["target_params"]
    feature_processing = config_dict["feature_processing"]
    source_processing = config_dict["source_processing"]
    target_processing = config_dict["target_processing"]
    algorithms = config_dict["algorithms"]
    initializer_params = config_dict["initializer_params"]
    refiner_params = config_dict["refiner_params"]
    options = config_dict["options"]

    # Enable verbose output
    if args.verbose or options["verbose"]:
        logger.setLevel(logging.DEBUG)
        set_logger_level(logging.DEBUG)
        print_config_dict(config_dict)

    # Instantiate initializer
    if algorithms["initializer"] is not None:
        logger.debug(f"Loading initializer {algorithms['initializer']}.")
        if not options["use_degrees"]:
            normal_angle_threshold = np.rad2deg(initializer_params["normal_angle_threshold"])
        else:
            normal_angle_threshold = initializer_params["normal_angle_threshold"]
        if algorithms['initializer'].lower() == "ransac":
            initializer = registration.RANSAC(max_iteration=initializer_params["max_iteration"],
                                              confidence=initializer_params["confidence"],
                                              max_correspondence_distance=initializer_params["max_correspondence_distance"],
                                              with_scaling=initializer_params["with_scaling"],
                                              ransac_n=initializer_params["ransac_n"],
                                              checkers=initializer_params["checkers"],
                                              similarity_threshold=initializer_params["similarity_threshold"],
                                              normal_angle_threshold=normal_angle_threshold)
        elif algorithms['initializer'].lower() == "fgr":
            initializer = registration.FastGlobalRegistration(max_iteration=initializer_params["max_iteration"],
                                                              max_correspondence_distance=initializer_params["max_correspondence_distance"])
        else:
            raise ValueError(f"Only `ransac` and `fgr` supported as `initializer` but is {algorithms['initializer']}.")

    # Instantiate refiner
    if algorithms["refiner"] is not None:
        logger.debug(f"Loading refiner {algorithms['refiner']}.")
        if algorithms["refiner"].lower() == "icp":
            refiner = registration.IterativeClosestPoint(relative_fitness=refiner_params["relative_fitness"],
                                                         relative_rmse=refiner_params["relative_rmse"],
                                                         max_iteration=refiner_params["max_iteration"],
                                                         max_correspondence_distance=refiner_params["max_correspondence_distance"],
                                                         estimation_method=refiner_params["estimation_method"],
                                                         with_scaling=refiner_params["with_scaling"],
                                                         kernel=refiner_params["kernel"],
                                                         kernel_noise_std=refiner_params["kernel_noise_std"])
        else:
            raise ValueError(f"Only `icp` supported as `refiner` but is {algorithms['refiner']}.")

    # Load source and target data
    logger.debug("Loading source data.")
    source_list = utils.eval_data_parallel(data_list=data["source_files"], **source_params)

    logger.debug("Loading target data.")
    target_list = utils.eval_data_parallel(data_list=data["target_files"], **target_params)

    # Process source and target data
    logger.debug("Processing source data.")
    source_list = utils.process_point_cloud_parallel(point_cloud_list=source_list,
                                                     downsample=source_processing["downsample"],
                                                     downsample_factor=source_processing["downsample_factor"],
                                                     remove_outlier=source_processing["remove_outlier"],
                                                     outlier_std_ratio=source_processing["outlier_std_ratio"],
                                                     scale=source_processing["scale"],
                                                     estimate_normals=source_processing["estimate_normals"],
                                                     recalculate_normals=source_processing["recalculate_normals"],
                                                     normalize_normals=source_processing["normalize_normals"],
                                                     orient_normals=source_processing["orient_normals"],
                                                     search_param=source_processing["search_param"],
                                                     search_param_knn=source_processing["search_param_knn"],
                                                     search_param_radius=source_processing["search_param_radius"],
                                                     draw=source_processing["draw"])

    logger.debug("Processing target data.")
    target_list = utils.process_point_cloud_parallel(point_cloud_list=target_list,
                                                     downsample=target_processing["downsample"],
                                                     downsample_factor=target_processing["downsample_factor"],
                                                     remove_outlier=target_processing["remove_outlier"],
                                                     outlier_std_ratio=target_processing["outlier_std_ratio"],
                                                     scale=target_processing["scale"],
                                                     estimate_normals=target_processing["estimate_normals"],
                                                     recalculate_normals=target_processing["recalculate_normals"],
                                                     normalize_normals=target_processing["normalize_normals"],
                                                     orient_normals=target_processing["orient_normals"],
                                                     search_param=target_processing["search_param"],
                                                     search_param_knn=target_processing["search_param_knn"],
                                                     search_param_radius=target_processing["search_param_radius"],
                                                     draw=target_processing["draw"])

    # Run registration algorithms
    results = list()
    init_list = data["init_poses"]
    if algorithms["initializer"] is not None:
        logger.debug(f"Running initializer {initializer.name}.")
        results = initializer.run_many(source_list=source_list,
                                       target_list=target_list,
                                       init_list=init_list,
                                       one_vs_one=data["one_vs_one"],
                                       n_times=initializer_params["n_times"],
                                       multi_scale=initializer_params["multi_scale"],
                                       source_scales=initializer_params["scales"],
                                       iterations=initializer_params["iterations"],
                                       radius_multiplier=initializer_params["radius_multiplier"],
                                       draw=initializer_params["draw"] or (args.draw and algorithms["refiner"] is None),
                                       overwrite_colors=initializer_params["overwrite_colors"],
                                       search_param=feature_processing["search_param"],
                                       search_param_knn=feature_processing["search_param_knn"],
                                       search_param_radius=feature_processing["search_param_radius"],
                                       progress=options["progress"] and not (args.verbose or options["verbose"]))
        init_list = [result.transformation for result in results]
    if algorithms["refiner"] is not None:
        logger.debug(f"Running refiner {refiner.name}.")
        results = refiner.run_many(source_list=source_list,
                                   target_list=target_list,
                                   init_list=init_list,
                                   one_vs_one=data["one_vs_one"],
                                   n_times=refiner_params["n_times"],
                                   multi_scale=refiner_params["multi_scale"],
                                   source_scales=refiner_params["scales"],
                                   iterations=refiner_params["iterations"],
                                   radius_multiplier=refiner_params["radius_multiplier"],
                                   crop_target_around_source=refiner_params["crop_target_around_source"],
                                   crop_scale=refiner_params["crop_scale"],
                                   draw=refiner_params["draw"] or args.draw,
                                   overwrite_colors=refiner_params["overwrite_colors"],
                                   progress=options["progress"] and not (args.verbose or options["verbose"]))
    logger.debug(f"Execution took {time.time() - start} seconds.")

    # Load ground truth data
    ground_truth = data["ground_truth"]
    if ground_truth is not None:
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        ground_truth = [utils.eval_transformation_data(gt) for gt in ground_truth]
        if len(ground_truth) == 1:
            ground_truth *= len(results)
        assert len(ground_truth) == len(results), f"'ground_truth' and 'results' must have equal length."

    # Evaluate registration results
    names = list()
    errors = list()
    if data["one_vs_one"]:
        assert len(results) == len(source_list) == len(target_list),\
            f"'results', 'source_list' and 'target_list' must have equal length."
        for i in range(len(results)):
            names.append(f"s{i} - t{i}")
            if ground_truth is not None:
                errors.append(utils.get_transformation_error(results[i].transformation,
                                                             ground_truth[i],
                                                             in_degrees=options["use_degrees"]))
            else:
                errors.append(('?', '?'))
    else:
        assert len(results) == len(source_list) * len(target_list),\
            f"'len(results)' must equal 'len(source_list) * len(target_list)."
        estimates = [T.transformation for T in results]
        for i in range(len(target_list)):
            for j in range(len(source_list)):
                names.append(f"s{j} - t{i}")
                if ground_truth is not None:
                    errors.append(utils.get_transformation_error(estimates.pop(0),
                                                                 ground_truth.pop(0),
                                                                 in_degrees=options["use_degrees"]))
                else:
                    errors.append(('?', '?'))
    errors_rot = [error[0] for error in errors]
    errors_trans = [error[1] for error in errors]

    # Print evaluation results
    if options["print_results"] or options["verbose"] or args.verbose:
        table = tabulate.tabulate([(name,
                                    result.fitness,
                                    result.inlier_rmse,
                                    len(result.correspondence_set),
                                    error_rot,
                                    error_trans) for name, result, error_rot, error_trans in zip(names,
                                                                                                 results,
                                                                                                 errors_rot,
                                                                                                 errors_trans)],
                                  headers=["source vs. target",
                                           "fitness",
                                           "inlier rmse",
                                           "# corresp.",
                                           f"error rot. {'[deg]' if options['use_degrees'] else '[rad]'}",
                                           "error trans. [m]"])
        print()
        print("RESULTS:\n=======")
        print(table)

    # Return results
    _return = options["return"].lower()
    return_data = dict()
    if "names" in _return:
        return_data["names"] = names
    if "results" in _return:
        return_data["results"] = results
    if "transformations" in _return:
        return_data["transformations"] = [result.transformation for result in results]
    if "errors_rot" in _return or "errors" in _return:
        return_data["errors_rot"] = errors_rot
    if "errors_trans" in _return or "errors" in _return:
        return_data["errors_trans"] = errors_trans
    return return_data


if __name__ == "__main__":
    run()
