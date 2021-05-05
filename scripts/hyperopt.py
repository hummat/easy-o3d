#!/usr/bin/env python3
"""Registration algorithms hyperparameter optimization.

Performs hyperparameter optimization for the registration algorithms from this package using Scikit Optimize
(requires `scikit-optimize` package). The optimization is performed on a list of source and target point clouds provided
as file paths and evaluated on ground truth transformations.
"""

import configparser
import time
import argparse
from typing import List, Union, Dict, Any
import skopt
import tabulate

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
                            if all(isinstance(v, list) for v in value):
                                space.append(skopt.space.Categorical(categories=[tuple(v) for v in value], name=key))
                        else:
                            space.append(skopt.space.Categorical(categories=value, name=key))
                    elif isinstance(value, tuple):
                        if any(isinstance(v, float) for v in value):
                            space.append(skopt.space.Real(low=value[0], high=value[1], name=key))
                        else:
                            space.append(skopt.space.Integer(low=value[0], high=value[1], name=key))
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
        for option, value in config_or_dict_from.items():
            for section in config_to.sections():
                if section not in sections_to_skip and option not in options_to_skip:
                    if add_missing_sections_and_keys:
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
    start = time.time()
    parser = argparse.ArgumentParser(description="Performs point cloud registration.")
    parser.add_argument("-c", "--config", default="hyperopt.ini", type=str, help="Path to hyperopt config.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Get verbose output during execution.")
    args = parser.parse_args()

    run_config = configparser.ConfigParser(inline_comment_prefixes='#')
    run_config.read("registration.ini")

    hyper_config = configparser.ConfigParser(inline_comment_prefixes='#')
    hyper_config.read(args.config)

    set_config_params_with_config_or_dict(config_or_dict_from=hyper_config,
                                          config_to=run_config)

    run_config["options"]["print_results"] = str(False)
    run_config["options"]["return"] = "everything"
    run_config["options"]["use_degrees"] = str(False)

    space = get_skopt_space_from_config(config=hyper_config)

    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params: Any):
        set_config_params_with_config_or_dict(config_or_dict_from=params,
                                              config_to=run_config,
                                              sections_to_skip=["DEFAULT", "optimization", "options", "data"])

        if args.verbose or hyper_config.getboolean("options", "verbose"):
            print_config(config=run_config)

        results = run(config=run_config)
        return sum(results.get("errors_rot") + results.get("errors_trans"))

    result = skopt.dummy_minimize(func=objective,
                                  dimensions=space,
                                  n_calls=hyper_config.getint("optimization", "iterations"),
                                  verbose=args.verbose or hyper_config.getboolean("options", "verbose"))


if __name__ == "__main__":
    main()
