"""Scripts for Easy Open3D.

Files:
    __init__.py: This file.
    hyperopt.ini: Initialization file for `hyperopt.py`.
    hyperopt.py: Registration algorithms hyperparameter optimization script.
    registration.ini: Initialization file for `run_registration.py`.
    run_registration.py: Performs point cloud registration using registration algorithms from this package.

Functions:
    hyperopt.get_skopt_space_from_config: Creates Scikit-Optimize search space from a ConfigParser object.
    hyperopt.set_config_params_with_config_or_dict: Sets ConfigParser object sections, options and values based on
                                                    another ConfigParser object or a dict.
    print_config: Pretty-print ConfigParser object content.
    run_registration.eval_config: Evaluates data types of a ConfigParser object.
    run_registration.print_config_dict: Pretty-prints a config dict created by 'eval_config'.
    run_registration.run: Runs the registration algorithms.
"""