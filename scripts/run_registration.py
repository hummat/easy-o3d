#!/usr/bin/env python3
"""Performs point cloud registration using registration algorithms from this package."""

from easy_o3d import utils, registration, set_logger_level
import numpy as np
import argparse
import logging
import time

logger = logging.getLogger(__name__)


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Performs point cloud registration.")
    parser.add_argument("--algorithms", nargs='+', type=str, choices=["icp", "ransac", "fgr"],
                        required=True, help="The registration algorithms used to align sources to the targets data.")
    parser.add_argument("--sources", nargs='+', type=str, required=True, help="List of paths to source data.")
    parser.add_argument("--targets", nargs='+', type=str, required=True, help="List of paths to target data.")
    parser.add_argument("--init", nargs='+', type=str, help="List of paths to initial source poses.")
    parser.add_argument("--verbose", action="store_true", help="Get verbose output during execution.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        set_logger_level(logging.DEBUG)

    refiner = None
    initializer = None
    for algorithm in args.algorithms:
        if algorithm.lower() == "icp":
            refiner = registration.IterativeClosestPoint()
        if algorithm.lower() == "ransac":
            initializer = registration.RANSAC()
        if algorithm.lower() == "fgr":
            initializer = registration.FastGlobalRegistration()

    sources = [utils.eval_data(data) for data in args.sources]
    targets = [utils.eval_data(data) for data in args.targets]
    if args.init:
        assert len(args.init) == len(sources) * len(targets)
        init_list = args.init
    else:
        init_list = [np.eye(4) for _ in range(len(sources) * len(targets))]

    results = list()
    for source in sources:
        for target in targets:
            if initializer is not None:
                result = initializer.run(source=source, target=target)
                init = result.transformation
            else:
                init = init_list.pop()
            if refiner is not None:
                result = refiner.run(source=source, target=target, init=init)
            results.append(result)

    logger.debug(f"Registration of {len(sources)} source(s) to {len(targets)} target(s) with initializer={initializer._name} and refiner={refiner._name} took {time.time() - start} seconds.")


if __name__ == "__main__":
    main()
