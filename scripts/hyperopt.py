#!/usr/bin/env python3
"""Registration hyperparameter optimization.

Performs hyperparameter optimization for the registration algorithms from this package using Scikit Optimize (requires `scikit-optimize` package).
The optimization is performed on a list of source and target point clouds provided as file paths and evaluated on ground thruth transformations.
"""

import configparser

import skopt
from run_registration import run


def main():
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read("registration.ini")
    config["options"]["verbose"] = str(False)
    run(config)


if __name__ == "__main__":
    main()
