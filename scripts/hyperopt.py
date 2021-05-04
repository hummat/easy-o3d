#!/usr/bin/env python3
"""Registration hyperparameter optimization.

Performs hyperparameter optimization for the registration algorithms from this package using Scikit Optimize (requires `scikit-optimize` package).
The optimization is performed on a list of source and target point clouds provided as file paths and evaluated on ground thruth transformations.
"""

import configparser
from typing import List, Union
import skopt
from run_registration import run


def get_skopt_space_from_config(config: configparser.ConfigParser) -> List[Union[skopt.space.Categorical,
                                                                                 skopt.space.Real]]:
    space = list()
    for section in config.sections():
        if section.lower() not in ["default", "optimization", "options", "data"]:
            for key, value in config.items(section):
                try:
                    value = eval(value)
                    if isinstance(value, list):
                        if key.lower() in ["checkers", "scales", "iterations", "radius_multiplier"]:
                            if isinstance(value[0], (list, tuple)):
                                space.append(skopt.space.Categorical(categories=[tuple(v) for v in value], name=key))
                        else:
                            space.append(skopt.space.Categorical(categories=value, name=key))
                    elif isinstance(value, tuple):
                        space.append(skopt.space.Real(low=value[0], high=value[1], name=key))
                except (NameError, SyntaxError):
                    pass
    return space


def main():
    run_config = configparser.ConfigParser(inline_comment_prefixes='#')
    run_config.read("registration.ini")

    hyper_config = configparser.ConfigParser(inline_comment_prefixes='#')
    hyper_config.read("hyperopt.ini")

    space = get_skopt_space_from_config(hyper_config)
    print(space)

    run_config["data"]["source_files"] = hyper_config["data"]["source_files"]
    run_config["data"]["target_files"] = hyper_config["data"]["target_files"]
    run_config["data"]["ground_truth"] = hyper_config["data"]["ground_truth"]
    run_config["data"]["init_poses"] = hyper_config["data"]["init_poses"]

    run_config["source_params"]["number_of_points"] = hyper_config["source_params"]["number_of_points"]
    run_config["target_params"]["number_of_points"] = hyper_config["target_params"]["number_of_points"]
    run_config["target_params"]["camera_intrinsic"] = hyper_config["target_params"]["camera_intrinsic"]
    run_config["target_params"]["camera_extrinsic"] = hyper_config["target_params"]["camera_extrinsic"]
    run_config["target_params"]["depth_trunc"] = str(5.0)

    #run_config["source_processing"]["estimate_normals"] = str(True)
    #run_config["target_processing"]["estimate_normals"] = str(True)
    #run_config["target_processing"]["compute_feature"] = str(True)

    run_config["algorithms"]["initializer"] = "none"

    run_config["initializer_params"]["draw"] = str(False)
    run_config["refiner_params"]["max_iteration"] = str(1)
    run_config["refiner_params"]["draw"] = str(False)

    run_config["options"]["print_results"] = str(False)
    run_config["options"]["use_degree"] = str(False)
    # run(run_config)


if __name__ == "__main__":
    main()
