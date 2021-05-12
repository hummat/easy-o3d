#!/usr/bin/env python3
"""Registration algorithms hyperparameter optimization script.

Performs hyperparameter optimization for the registration algorithms from this package using Scikit Optimize
(requires `scikit-optimize` package). The optimization is performed on a list of source and target point clouds provided
as file paths and evaluated on ground truth transformations.

Functions:
    get_skopt_space_from_config: Creates Scikit-Optimize search space from a ConfigParser object.
    set_config_params_with_config_or_dict: Sets ConfigParser object sections, options and values based on another
                                           ConfigParser object or a dict.
    print_config: Pretty-print ConfigParser object content.
"""
import configparser
import os
import sys
import time
import argparse
from typing import List, Union, Dict, Any, Tuple
import logging
import hashlib
import copy
import random

import tqdm
import numpy as np
import skopt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
import tabulate

from .run_registration import run

logger = logging.getLogger(__name__)


def get_skopt_space_from_config(config: configparser.ConfigParser) -> List[Union[skopt.space.Categorical,
                                                                                 skopt.space.Real,
                                                                                 skopt.space.Integer]]:
    """Creates Scikit-Optimize search space from a ConfigParser object.

    Args:
        config: The ConfigParser object specifying the options and their bounds to optimize.

    Returns:
        The Scikit-Optimize search space as a list of sample spaces.
    """
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

    optimizer = config.get("optimization", "optimizer")
    for section in config.sections():
        if section.lower() not in sections_to_skip:
            for option, value in config.items(section):
                if section not in options_to_skip or option not in options_to_skip[section.lower()]:
                    try:
                        name = ': '.join([section, option])
                        if 'e' in value:
                            if isinstance(eval(value), tuple) and all(isinstance(v, (float, int)) for v in eval(value)):
                                prior = "log-uniform"
                            else:
                                prior = "uniform"
                        else:
                            prior = "uniform"
                        value = eval(value)

                        if isinstance(value, list):
                            if option.lower() in ["checkers", "scales", "iterations", "radius_multiplier"]:
                                if optimizer in ["rnd", "rand", "random"]:
                                    if all(isinstance(v, list) for v in value):
                                        space.append(skopt.space.Categorical(categories=[tuple(v) for v in value],
                                                                             name=name))
                            else:
                                space.append(skopt.space.Categorical(categories=value, name=name))
                        elif isinstance(value, tuple):
                            if any(isinstance(v, float) for v in value):
                                space.append(skopt.space.Real(low=value[0],
                                                              high=value[1],
                                                              prior=prior,
                                                              name=name))
                            else:
                                space.append(skopt.space.Integer(low=value[0],
                                                                 high=value[1],
                                                                 prior=prior,
                                                                 name=name))
                    except (NameError, SyntaxError):
                        pass
    return space


def set_config_params_with_config_or_dict(config_or_dict_from: Union[configparser.ConfigParser, Dict[str, Any]],
                                          config_to: configparser.ConfigParser,
                                          sections_to_skip: List[str] = list(),
                                          options_to_skip: List[str] = list(),
                                          add_missing_sections_and_options: bool = False) -> None:
    """Sets ConfigParser object sections, options and values based on another ConfigParser object or a dict.

    Args:
        config_or_dict_from: The ConfigParser object or dict to read from.
        config_to: The ConfigParser object to write to.
        sections_to_skip: Sections in 'config_or_dict_from' that shouldn't be written to 'config_to'.
        options_to_skip: Options in 'config_or_dict_from' sections that shouldn't be written to 'config_to'.
        add_missing_sections_and_options: Sections and/or options only present in 'config_or_dict_from' are added to
                                          'config_to'.
    """
    if isinstance(config_or_dict_from, configparser.ConfigParser):
        for section in config_or_dict_from.sections():
            if section not in sections_to_skip:
                for option, value in config_or_dict_from.items(section):
                    if option not in options_to_skip:
                        if add_missing_sections_and_options:
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
                        option = option.replace(section + ': ', '')
                        if option not in options_to_skip:
                            value = getattr(value, "tolist", lambda: value)()
                            if add_missing_sections_and_options:
                                if not config_to.has_section(section):
                                    config_to.add_section(section)
                                config_to[section][option] = str(value)
                            else:
                                if config_to.has_option(section, option):
                                    config_to[section][option] = str(value)
    else:
        raise TypeError(f"Input must be of type `dict` or `ConfigParser`, not {type(config_or_dict_from)}.")


def print_config(config: configparser.ConfigParser, pretty: bool = True) -> None:
    """Pretty-print ConfigParser object content.

    Args:
        config: The ConfigParser object.
        pretty: Pretty-print ConfigParser object sections and options.
    """
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
    cd = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Performs point cloud registration.")
    parser.add_argument("-c", "--config", default=os.path.join(cd, "hyperopt.ini"), type=str,
                        help="/path/to/hyperopt.ini.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Get verbose output during execution.")
    parser.add_argument("-d", "--draw", action="store_true", help="Visualize results results.")
    parser.add_argument("-o", "--output", default=os.path.join(cd, "output"), type=str, help="/path/to/output/dir")
    args = parser.parse_args()

    # Read hyperparameter optimization config (hyper config) from argument
    hyper_config = configparser.ConfigParser(inline_comment_prefixes='#')
    hyper_config.read(args.config)

    # Set frequent config options
    optimization = hyper_config["optimization"]
    optimizer = optimization.get("optimizer").lower()
    options = hyper_config["options"]

    # Read config from argument or file
    run_config = configparser.ConfigParser(inline_comment_prefixes='#')
    run_config.read(os.path.join(cd, "registration.ini"))

    # Enable verbose output
    verbose = args.verbose or options.getboolean("verbose")
    if verbose:
        logger.setLevel(logging.DEBUG)
        print_config(hyper_config)

    # Set run config with values from hyper config
    set_config_params_with_config_or_dict(config_or_dict_from=hyper_config, config_to=run_config)
    run_config["options"]["verbose"] = str(False)
    run_config["options"]["progress"] = str(False)
    run_config["options"]["print_results"] = str(False)
    run_config["options"]["return"] = "results+errors" if "error" in optimization.get("minimize").lower() else "results"
    run_config["options"]["use_degrees"] = str(False)
    configs = list()

    # Create a Scikit-Optimize search space from the hyper config
    space = get_skopt_space_from_config(config=hyper_config)

    # Make a progress bar
    progress = tqdm.tqdm(range(optimization.getint("n_calls")),
                         desc=f"Optimization ({optimizer.upper()})",
                         file=sys.stdout,
                         disable=not options.getboolean("progress") or verbose)

    def progress_callback(_result: skopt.utils.OptimizeResult) -> None:
        progress.set_postfix_str(f"Current minimum ({optimization.get('minimize')}): {np.min(_result.func_vals)}")
        progress.update()

    # Make a checkpointer
    def checkpoint_callback(_result: skopt.utils.OptimizeResult) -> None:
        try:
            if len(_result.x_iters) % 10 == 1:
                _res = copy.deepcopy(_result)
                del _res.specs['args']['callback']
                del _res.specs['args']['func']
                skopt.dump(res=_res, filename=os.path.join(args.output, "checkpoint.pkl"), compress=True)
        except Exception as ex:
            message = f"Couldn't save checkpoint due to exception: {ex}. Skipping."
            logger.exception(message) if progress.disable else progress.write(message)

    # The optimization objective function
    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params: Dict[str, Any]) -> Union[float, Tuple[float, float]]:
        start = time.time()

        if verbose:
            print(tabulate.tabulate({"Option": list(params.keys()), "Values": list(params.values())},
                                    headers="keys",
                                    missingval="None"))

        set_config_params_with_config_or_dict(config_or_dict_from=params,
                                              config_to=run_config,
                                              sections_to_skip=["DEFAULT", "optimization", "options", "data"])
        configs.append(copy.deepcopy(run_config))

        try:
            results = run(config=run_config)

            if any("error" in key for key in results.keys()):
                if optimization.get("minimize").lower() in ["errors", "errors_rot+errors_trans"]:
                    cost = sum(results["errors_rot"] + results["errors_trans"])
                elif optimization.get("minimize").lower() == "errors_rot":
                    cost = sum(results["errors_rot"])
                elif optimization.get("minimize").lower() == "errors_trans":
                    cost = sum(results["errors_trans"])
                else:
                    raise ValueError(f"Invalid config option '{optimization.get('minimize')}' for 'minimize'.")
            else:
                if optimization.get("minimize").lower() in ["errors", "errors_rot+errors_trans"]:
                    message = f"Optimization objective is '{optimization.get('minimize')}' which was not found in" \
                              f"returned results. Did you provide ground truth data? Using inlier RMSE instead."
                    raise ValueError(message)
                elif optimization.get("minimize").lower() == "inlier_rmse-fitness":
                    cost = sum([r.inlier_rmse - r.fitness for r in results["results"]])
                elif optimization.get("minimize").lower() in ["fitness", "-fitness"]:
                    cost = -sum([r.fitness for r in results["results"]])
                elif optimization.get("minimize").lower() == "inlier_rmse":
                    cost = sum([r.inlier_rmse for r in results["results"]])
                else:
                    raise ValueError(f"Invalid config option '{optimization.get('minimize')}' for 'minimize'.")
                if optimization.getboolean("scale_objective"):
                    num_correspondences = sum([len(r.correspondence_set) for r in results["results"]])
                    cost = cost / num_correspondences if num_correspondences > 0 else 1e30
        except Exception as ex:
            message = f"Caught exception during execution of 'run_registration.run': {ex}"
            logger.exception(message) if progress.disable else progress.write(message)
            cost = 1e30

        if not np.isfinite(cost):
            cost = 1e30
        if optimizer not in ["rnd", "random"] and optimization.get("acq_func").lower() in ["eips", "pips"]:
            return cost, time.time() - start
        else:
            return cost

    # Load initial points
    init = optimization.get("init_from_file_or_list")
    x0, y0 = None, None
    try:
        init = eval(init)
        if init is not None and isinstance(init, (list, tuple)):
            x0 = init
            y0 = None
    except (NameError, SyntaxError):
        try:
            init = skopt.load(init)
            x0 = init.x_iters
            y0 = init.func_vals
        except FileNotFoundError:
            logger.exception(f"Couldn't load initial points from {init}. Skipping.")
            x0 = None
            y0 = None

    # Run optimization
    if optimizer in ["rnd", "rand", "random"]:
        result = skopt.dummy_minimize(func=objective,
                                      dimensions=space,
                                      n_calls=optimization.getint("n_calls"),
                                      initial_point_generator=optimization.get("initial_point_generator"),
                                      x0=x0,
                                      y0=y0,
                                      random_state=eval(optimization.get("random_state")),
                                      verbose=verbose,
                                      callback=[progress_callback, checkpoint_callback])
    elif all(c in optimizer for c in ['r', 'f']):
        result = skopt.forest_minimize(func=objective,
                                       dimensions=space,
                                       base_estimator=optimization.get("base_estimator").upper(),
                                       n_calls=optimization.getint("n_calls"),
                                       n_initial_points=optimization.getint("n_initial_points"),
                                       acq_func=optimization.get("acq_func"),
                                       initial_point_generator=optimization.get("initial_point_generator"),
                                       x0=x0,
                                       y0=y0,
                                       random_state=eval(optimization.get("random_state")),
                                       verbose=verbose,
                                       callback=[progress_callback, checkpoint_callback],
                                       n_points=optimization.getint("n_points"),
                                       xi=optimization.getfloat("xi"),
                                       kappa=optimization.getfloat("kappa"),
                                       n_jobs=optimization.getint("n_jobs"))
    elif all(c in optimizer for c in ['g', 'p']):
        noise = optimization.get("noise").lower()
        noise = noise if "gauss" in noise else optimization.getfloat("noise")
        result = skopt.gp_minimize(func=objective,
                                   dimensions=space,
                                   n_calls=optimization.getint("n_calls"),
                                   n_initial_points=optimization.getint("n_initial_points"),
                                   acq_func=optimization.get("acq_func"),
                                   acq_optimizer=optimization.get("acq_optimizer"),
                                   initial_point_generator=optimization.get("initial_point_generator"),
                                   x0=x0,
                                   y0=y0,
                                   random_state=eval(optimization.get("random_state")),
                                   verbose=verbose,
                                   callback=[progress_callback, checkpoint_callback],
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
                                     x0=x0,
                                     y0=y0,
                                     random_state=eval(optimization.get("random_state")),
                                     verbose=verbose,
                                     callback=[progress_callback, checkpoint_callback],
                                     n_points=optimization.getint("n_points"),
                                     xi=optimization.getfloat("xi"),
                                     kappa=optimization.getfloat("kappa"),
                                     n_jobs=optimization.getint("n_jobs"))
    else:
        raise ValueError(f"Invalid optimizer {optimizer}.")
    progress.close()

    # Print results
    names = None
    dims = None
    try:
        res = np.concatenate([np.expand_dims(result.func_vals, axis=1), result.x_iters], axis=1)
        iters = np.argsort(res[:, 0])
        table_values = res[iters]
        names = [dim.name for dim in space]
        dims = [f"X_{i}" for i in range(len(names))] if len(names) > 10 else names
        table_headers = ["Iter", "Cost"] + dims
        print()
        print("OPTIMIZATION RESULTS:\n====================")
        print(tabulate.tabulate(table_values, headers=table_headers, missingval="None", showindex=list(iters)))
        if len(names) > 10:
            print()
            print("DIMENSIONS <-> NAMES:\n====================")
            print(tabulate.tabulate({"Dimension": dims, "Name": names}, headers="keys"))
    except Exception as e:
        logger.exception(f"Printing of results failed due to exception: {e}. Trying to save results.")

    # Visualize results
    if (args.draw or options.getboolean("draw")) and names is not None:
        try:
            if len(names) > 5:
                logger.info(f"Drawing visualizations for upt to 10 dimensions. This can take some time.")
            from matplotlib import pyplot as plt
            plot_convergence(result)
            try:
                plot_dims = eval(options.get("plot_dims"))
                if isinstance(plot_dims, float):
                    plot_dims = int(plot_dims)
                elif isinstance(plot_dims, int) or plot_dims is None:
                    pass
                else:
                    raise TypeError(f"'plot_dims' must by None or int but is {type(plot_dims)}. Skipping selection.")
            except (NameError, SyntaxError, TypeError):
                plot_dims = None
            if (plot_dims is None and len(names) > 10) or (isinstance(plot_dims, int) and plot_dims > 10):
                logger.info(f"Too many dimensions ({len(names)}) for 'evaluations' and 'objective' plots. "
                            f"Only drawing the first 10.")
                plot_dims = 10
            if isinstance(plot_dims, int):
                plot_dims = names[:plot_dims]
            else:
                plot_dims = None
            plot_evaluations(result, dimensions=dims if plot_dims is None else plot_dims, plot_dims=plot_dims)
            if optimizer not in ["rnd", "rand", "random"]:
                plot_objective(result, dimensions=dims if plot_dims is None else plot_dims, plot_dims=plot_dims)
            else:
                logger.info(f"Can't plot 'objective' for optimizer '{optimizer}'.")
            plt.show()
        except Exception as e:
            logger.exception(f"Drawing failed due to exception: {e}. Skipping.")

    # Save best result config
    filename = None
    config_hash = None
    if options.getboolean("save") or names is None:
        try:
            filename = options.get("filename")
            try:
                filename = eval(filename)
            except (NameError, SyntaxError):
                if filename.lower() == "none":
                    filename = None
            best_config = configs[np.argmin(result.func_vals)]
            best_config["options"]["progress"] = str(True)
            best_config["options"]["print_results"] = str(True)
            if filename is None:
                if options.getboolean("overwrite"):
                    config_hash = str().join([value for option, value in hyper_config.items("data")])
                else:
                    config_hash = str()
                    for section in hyper_config.sections():
                        for option, value in hyper_config.items(section):
                            config_hash += value
                config_hash = hashlib.md5(config_hash.encode('utf-8')).hexdigest()
                output_path = os.path.join(args.output, f"{config_hash}.ini")
            else:
                output_path = os.path.join(args.output, f"{filename.split('.')[0]}.ini")
            with open(output_path, 'w') as configfile:
                best_config.write(configfile)
        except Exception as e:
            config_hash = hex(random.getrandbits(128))
            logger.exception(f"Saving of best result failed due to exception: {e}. Trying to dump all results.")

        # Save results
        if filename is None:
            output_path = os.path.join(args.output, f"{config_hash}.pkl")
        else:
            output_path = os.path.join(args.output, f"{filename.split('.')[0]}.pkl")
        try:
            del result.specs['args']['callback']
            skopt.dump(res=result, filename=output_path, compress=True)
        except Exception as e:
            logger.debug(f"Caught exception during 'skopt.dump': {e}")
            logger.debug("Trying to store the result without the objective.")
            skopt.dump(res=result, filename=output_path, store_objective=False, compress=True)
        finally:
            logger.debug("Deleting the objective.")
            del result.specs['args']['func']
            skopt.dump(res=result, filename=output_path, compress=True)


if __name__ == "__main__":
    main()
