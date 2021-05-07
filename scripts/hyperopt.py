#!/usr/bin/env python3
"""Registration algorithms hyperparameter optimization.

Performs hyperparameter optimization for the registration algorithms from this package using Scikit Optimize
(requires `scikit-optimize` package). The optimization is performed on a list of source and target point clouds provided
as file paths and evaluated on ground truth transformations.
"""

import configparser
import os
import time
import argparse
from typing import List, Union, Dict, Any, Tuple
import logging
import hashlib

import tqdm
import numpy as np
import skopt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
import tabulate

from run_registration import run

logger = logging.getLogger(__name__)


def get_skopt_space_from_config(config: configparser.ConfigParser) -> List[Union[skopt.space.Categorical,
                                                                                 skopt.space.Real,
                                                                                 skopt.space.Integer]]:
    space = list()
    sections_to_skip = ["default", "optimization", "options", "data"]
    options_to_skip = dict()
    for prefix in ["feature", "source", "target"]:
        section = prefix + "_processing"
        search_param = config.get(section, "search_param")
        try:
            if search_param.lower() == "none" or eval(search_param) is None:
                options_to_skip[section] = ["search_param_knn", "search_param_radius"]
        except (NameError, SyntaxError):
            pass
        if prefix in ["source", "target"]:
            downsample = config.get(section, "downsample")
            try:
                if downsample.lower() == "none" or eval(downsample) is None:
                    options_to_skip[section] = ["downsample_factor"]
            except (NameError, SyntaxError):
                pass
    try:
        initializer = config.get("algorithms", "initializer")
        if initializer.lower() == "none" or eval(initializer) is None:
            sections_to_skip.append("initializer_params")
    except (NameError, SyntaxError):
        pass
    try:
        refiner = config.get("algorithms", "refiner")
        if refiner.lower() == "none" or eval(refiner) is None:
            sections_to_skip.append("refiner_params")
    except (NameError, SyntaxError):
        pass

    for section in config.sections():
        if section.lower() not in sections_to_skip:
            for option, value in config.items(section):
                if section not in options_to_skip or option not in options_to_skip[section.lower()]:
                    try:
                        prior = "log-uniform" if 'e' in value and isinstance(eval(value), (float, int)) else "uniform"
                        value = eval(value)

                        if isinstance(value, list):
                            if option.lower() in ["checkers", "scales", "iterations", "radius_multiplier"]:
                                if all(isinstance(v, list) for v in value):
                                    space.append(skopt.space.Categorical(categories=[tuple(v) for v in value],
                                                                         name=section + option))
                            else:
                                space.append(skopt.space.Categorical(categories=value, name=section + option))
                        elif isinstance(value, tuple):
                            if any(isinstance(v, float) for v in value):
                                space.append(skopt.space.Real(low=value[0],
                                                              high=value[1],
                                                              prior=prior,
                                                              name=section + option))
                            else:
                                space.append(skopt.space.Integer(low=value[0],
                                                                 high=value[1],
                                                                 prior=prior,
                                                                 name=section + option))
                    except (NameError, SyntaxError):
                        pass
    return space


def set_config_params_with_config_or_dict(config_or_dict_from: Union[configparser.ConfigParser, Dict[str, Any]],
                                          config_to: configparser.ConfigParser,
                                          sections_to_skip: List[str] = list(),
                                          options_to_skip: List[str] = list(),
                                          add_missing_sections_and_keys: bool = False) -> None:
    if isinstance(config_or_dict_from, configparser.ConfigParser):
        for section in config_or_dict_from.sections():
            if section not in sections_to_skip:
                for option, value in config_or_dict_from.items(section):
                    if option not in options_to_skip:
                        if add_missing_sections_and_keys:
                            config_to[section][option] = config_or_dict_from.get(section, option)
                        else:
                            if section in config_to.sections():
                                if option in config_to[section]:
                                    config_to[section][option] = config_or_dict_from.get(section, option)
    elif isinstance(config_or_dict_from, dict):
        for section in config_to.sections():
            if section not in sections_to_skip:
                for option, value in config_or_dict_from.items():
                    if section in option:
                        option = option.replace(section, '')
                        if option not in options_to_skip:
                            value = getattr(value, "tolist", lambda: value)()
                            if add_missing_sections_and_keys:
                                if not config_to.has_section(section):
                                    config_to.add_section(section)
                                config_to[section][option] = str(value)
                            else:
                                if config_to.has_option(section, option):
                                    config_to[section][option] = str(value)
    else:
        raise TypeError(f"Input must be of type `dict` or `ConfigParser`, not {type(config_or_dict_from)}.")


def print_config(config: configparser.ConfigParser, pretty: bool = True) -> None:
    config_list = list()
    for section in config.sections():
        config_list.append(("", ""))
        config_list.append((section.upper().replace('_', ' ') if pretty else section, ""))
        config_list.append(('-' * len(section), ""))
        for key, value in config.items(section):
            config_list.append((key.capitalize().replace('_', ' ') if pretty else key,
                                value.capitalize() if value.lower() in ["true", "false", "none"] and pretty else value))
    print(tabulate.tabulate(config_list))


def main():
    # Evaluate command line arguments
    parser = argparse.ArgumentParser(description="Performs point cloud registration.")
    parser.add_argument("-c", "--config", default="hyperopt.ini", type=str, help="/path/to/hyperopt.ini.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Get verbose output during execution.")
    parser.add_argument("-d", "--draw", action="store_true", help="Visualize hyperopt results.")
    parser.add_argument("-o", "--output", default=os.path.dirname(os.path.abspath(__file__)), type=str,
                        help="/path/to/output/dir")
    args = parser.parse_args()

    # Read hyperparameter optimization config (hyper config) from argument
    hyper_config = configparser.ConfigParser(inline_comment_prefixes='#')
    hyper_config.read(args.config)

    # Evaluate config
    optimization = hyper_config["optimization"]
    optimizer = optimization.get("optimizer").lower()
    options = hyper_config["options"]

    # Read config from argument or file
    run_config = configparser.ConfigParser(inline_comment_prefixes='#')
    run_config.read("registration.ini")

    # Enable verbose output
    if args.verbose or options.getboolean("verbose"):
        logger.setLevel(logging.DEBUG)

    # Set run config with values from hyper config
    set_config_params_with_config_or_dict(config_or_dict_from=hyper_config,
                                          config_to=run_config)
    run_config["options"]["print_results"] = str(False)
    run_config["options"]["return"] = "results+errors"
    run_config["options"]["use_degrees"] = str(False)

    # Create a Scikit-Optimize search space from the hyper config
    space = get_skopt_space_from_config(config=hyper_config)
    configs = list()

    # Creates pretty parameter names
    @skopt.utils.use_named_args(dimensions=space)
    def get_param_names_from_skopt_space(**params: Dict[str, Any]) -> List[str]:
        _names = list(params.keys())
        for _section in run_config.sections():
            for i, _name in enumerate(_names):
                if _section in _name:
                    _names[i] = _name.replace(_section, _section + ': ')
        return _names

    # The optimization objective function
    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params: Dict[str, Any]) -> Union[float, Tuple[float, float]]:
        start = time.time()

        if args.verbose or options.getboolean("verbose"):
            print(tabulate.tabulate(np.expand_dims(list(params.values()), axis=0),
                                    headers=get_param_names_from_skopt_space(space)))

        set_config_params_with_config_or_dict(config_or_dict_from=params,
                                              config_to=run_config,
                                              sections_to_skip=["DEFAULT", "optimization", "options", "data"])
        configs.append(run_config)

        try:
            results = run(config=run_config)

            if any("errors" in key for key in results.keys()):
                if optimization.get("minimize").lower() in ["errors", "errors_rot+errors_trans"]:
                    cost = sum(results["errors_rot"] + results["errors_trans"])
                elif optimization.get("minimize").lower() == "errors_rot":
                    cost = sum(results["errors_rot"])
                elif optimization.get("minimize").lower() == "errors_trans":
                    cost = sum(results["errors_trans"])
                else:
                    raise ValueError(f"Invalid option '{optimization.get('minimize')}' for 'minimize'.")
            else:
                if optimization.get("minimize").lower() in ["errors", "errors_rot+errors_trans"]:
                    logger.warning(f"Optimization objective is '{optimization.get('minimize')}' which was not found in "
                                   f"returned results. Did you provide ground truth data? Using inlier RMSE instead.")
                    cost = sum([r.inlier_rmse for r in results["results"]])
                elif optimization.get("minimize").lower() == "inlier_rmse-fitness":
                    cost = sum([r.inlier_rmse - r.fitness for r in results["results"]])
                elif optimization.get("minimize").lower() in ["fitness", "-fitness"]:
                    cost = -sum([r.fitness for r in results["results"]])
                elif optimization.get("minimize").lower() == "inlier_rmse":
                    cost = sum([r.inlier_rmse for r in results["results"]])
                else:
                    raise ValueError(f"Invalid option '{optimization.get('minimize')}' for 'minimize'.")
        except Exception as ex:
            logger.warning(f"Caught exception during 'run': {ex}")
            cost = 1e30

        if np.isnan(cost) or np.isinf(cost) or not np.isfinite(cost):
            cost = 1e30
        if optimizer not in ["rnd", "random"] and optimization.get("acq_func").lower() in ["eips", "pips"]:
            return cost, time.time() - start
        else:
            return cost

    # Run optimization
    progress = tqdm.tqdm(range(optimization.getint("n_calls")), disable=args.verbose or options.getboolean("verbose"))

    def progress_update(_result: Any):
        progress.update()

    if optimizer in ["rnd", "random"]:
        result = skopt.dummy_minimize(func=objective,
                                      dimensions=space,
                                      n_calls=optimization.getint("n_calls"),
                                      initial_point_generator=optimization.get("initial_point_generator"),
                                      random_state=eval(optimization.get("random_state")),
                                      verbose=args.verbose or options.getboolean("verbose"),
                                      callback=progress_update)
    elif all(c in optimizer for c in ['r', 'f']):
        result = skopt.forest_minimize(func=objective,
                                       dimensions=space,
                                       base_estimator=optimization.get("base_estimator").upper(),
                                       n_calls=optimization.getint("n_calls"),
                                       n_initial_points=optimization.getint("n_initial_points"),
                                       acq_func=optimization.get("acq_func"),
                                       initial_point_generator=optimization.get("initial_point_generator"),
                                       random_state=eval(optimization.get("random_state")),
                                       verbose=args.verbose or options.getboolean("verbose"),
                                       callback=progress_update,
                                       n_points=optimization.getint("n_points"),
                                       xi=optimization.getfloat("xi"),
                                       kappa=optimization.getfloat("kappa"),
                                       n_jobs=optimization.getint("n_jobs"))
    elif all(c in optimizer for c in ['g', 'p']):
        noise = optimization.get("noise").lower() if "gauss" in optimization.get(
            "noise").lower() else optimization.getfloat("noise")
        result = skopt.gp_minimize(func=objective,
                                   dimensions=space,
                                   n_calls=optimization.getint("n_calls"),
                                   n_initial_points=optimization.getint("n_initial_points"),
                                   acq_func=optimization.get("acq_func"),
                                   acq_optimizer=optimization.get("acq_optimizer"),
                                   initial_point_generator=optimization.get("initial_point_generator"),
                                   random_state=eval(optimization.get("random_state")),
                                   verbose=args.verbose or options.getboolean("verbose"),
                                   callback=progress_update,
                                   n_points=optimization.getint("n_points"),
                                   n_restarts_optimizer=optimization.getint("n_restarts_optimizer"),
                                   xi=optimization.getfloat("xi"),
                                   kappa=optimization.getfloat("kappa"),
                                   noise=noise,
                                   n_jobs=optimization.getint("n_jobs"))
    elif all(c in optimizer for c in ['g', 'b', 'r', 't']):
        result = skopt.gbrt_minimize(func=objective,
                                     dimensions=space,
                                     n_calls=optimization.getint("n_calls"),
                                     n_initial_points=optimization.getint("n_initial_points"),
                                     acq_func=optimization.get("acq_func"),
                                     initial_point_generator=optimization.get("initial_point_generator"),
                                     random_state=eval(optimization.get("random_state")),
                                     verbose=args.verbose or options.getboolean("verbose"),
                                     callback=progress_update,
                                     n_points=optimization.getint("n_points"),
                                     xi=optimization.getfloat("xi"),
                                     kappa=optimization.getfloat("kappa"),
                                     n_jobs=optimization.getint("n_jobs"))
    else:
        raise ValueError(f"Invalid optimizer {optimizer}.")

    # Print results
    res = np.concatenate([np.expand_dims(result.func_vals, axis=1), result.x_iters], axis=1)
    iters = np.argsort(res[:, 0])
    table_values = res[iters]
    names = get_param_names_from_skopt_space(space)
    dims = [f"X_{i}" for i in range(len(names))] if len(names) > 10 else names
    table_headers = ["Iter", "Cost"] + dims
    print()
    print("OPTIMIZATION RESULTS:\n=====================")
    print(tabulate.tabulate(table_values, headers=table_headers, missingval="None", showindex=list(iters)))
    if len(names) > 10:
        print()
        print("DIMENSIONS <-> NAMES:\n=====================")
        print(tabulate.tabulate({"Dimension": dims, "Name": names}, headers="keys"))

    # Visualize results
    if args.draw or options.getboolean("draw"):
        if len(names) > 5:
            logger.info("Drawing visualizations for {len(names)} dimensions. This can take some time.")
        from matplotlib import pyplot as plt
        plot_convergence(result)
        plot_evaluations(result, dimensions=dims)
        if optimizer not in ["rnd", "random"]:
            plot_objective(result, dimensions=dims)
        plt.show()

    if options.getboolean("save"):
        # Save best result config
        if options.getboolean("overwrite"):
            config_hash = str().join([value for option, value in hyper_config.items("data")])
        else:
            config_hash = str()
            for section in hyper_config.sections():
                for option, value in hyper_config.items(section):
                    config_hash += value
        config_hash = hashlib.md5(config_hash.encode('utf-8')).hexdigest()
        best_config = configs[np.argmin(result.func_vals)]

        best_config["options"]["print_results"] = str(True)
        output_path = os.path.join(args.output, f"best_result_{config_hash}.ini")
        with open(output_path, 'w') as configfile:
            best_config.write(configfile)

        # Save results
        output_path = os.path.join(args.output, f"hyperopt_results_{config_hash}.pkl")
        try:
            del result.specs['args']['callback']
            skopt.dump(res=result, filename=output_path)
        except Exception as e:
            logger.debug(f"Caught exception during 'skopt.dump': {e}")
            logger.debug("Trying to store the result without the objective.")
            skopt.dump(res=result, filename=output_path, store_objective=False)
        finally:
            logger.debug("Deleting the objective.")
            del result.specs['args']['func']
            skopt.dump(res=result, filename=output_path)


if __name__ == "__main__":
    main()
