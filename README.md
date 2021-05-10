# Easy Open3D
Welcome to Easy Open3D the easy-to-use wrapper around (as well as utility functions and scripts for) some of
[Open3D](http://www.open3d.org) 's registration functionality.

## What is registration?
In 3D data analysis, the term _registration_ usually refers to the process of aligning two partly overlapping point
clouds such that the result is a merged version of the input. It is frequently used to obtain a 3D scan of a large scene
by stitching together several smaller parts.
In robotics however, we can use the same algorithms to register a 3D model of an object (called `source`) we want to
manipulate to the part of the 3D scan (obtained through depth sensors; called `target`) of the scene the robot operates
in to find its pose in this scene. The pose of an object consists of its rotation and translation in some coordinate
frame, typically the robots head frame, camera frame or world frame.

### A simple (yet slightly contrived) example
Say we want to find the pose of Suzanne (the blue ape head on the chair who is the Blender mascot) in this scene:

![](./tests/test_data/bop_data/obj_of_interest/train_pbr/000000/rgb/000020.png)

We are also given a 3D model of Suzanne (the `source`) and a depth image of the scene (the `target`):

{% include test_data.html %}

The goal is to rotate and move the `source` around until it matches the position and rotation of the corresponding
points the `target`. A successful registration looks like this:

{% include registration_result.html %}

The `source` (red) and `target` (gray) point clouds overlap, indicating a tight alignment.

## Highlights
1. High-level wrappers around
   [global pose estimation](https://github.com/hummat/easy-o3d/blob/22f760c46450b1e6e4a595ae9d79aa6846c0cda6/easy_o3d/registration.py#L208) and
   [pose refinement](https://github.com/hummat/easy-o3d/blob/22f760c46450b1e6e4a595ae9d79aa6846c0cda6/easy_o3d/registration.py#L65).
2. Seamless and flexible
   [data loading](https://github.com/hummat/easy-o3d/blob/22f760c46450b1e6e4a595ae9d79aa6846c0cda6/easy_o3d/utils.py#L92)
   from various input formats as well as further
   [utility functionality](https://github.com/hummat/easy-o3d/blob/master/easy_o3d/utils.py).
3. Flexible, fast and easy-to-use [scripts](https://github.com/hummat/easy-o3d/tree/master/scripts) to obtain the 6D
   pose of multiple objects from multiple scenes including
   [hyperparameter optimization](https://github.com/hummat/easy-o3d/blob/master/scripts/hyperopt.py).

## Installation
The simplest way to install the package is to clone the repository to a local directory:
```commandline
git clone https://github.com/hummat/easy-o3d
```
and use Pythons package manager `pip` to install it into your current Python environment:
```commandline
pip install /path/to/easy-o3d/repository/clone
```
To use the [hyperparameter optimization script](https://github.com/hummat/easy-o3d/blob/master/scripts/hyperopt.py),
also install the optional dependency `scikit-optimize`:
```commandline
pip install /path/to/easy-o3d/repository/clone[hyper]
```
(or manually with `pip install scikit-optimize`).

Alternatively, you can install the [required packages](https://github.com/hummat/easy-o3d/blob/master/requirements.txt)
manually and directly import the code from your local clone of the repository:
```commandline
pip install /path/to/requirements.txt
```
The required packages (and their version used during development) are:
1. `numpy == 1.20.2`
2. `open3d == 0.12.0`
3. (`scikit-optimize == 0.8.1`)
```commandline
pip install numpy==1.20.2 open3d==0.12.0 scikit-optimize==0.8.1
```

## Usage
Fundamentally, there are two ways to use this project. If you only want to find the pose of some objects in some scenes,
simply throw them at the
[`registration.py`](https://github.com/hummat/easy-o3d/tree/master/scripts/registration.py) script
(and potentially run the [`hyperopt.py`](https://github.com/hummat/easy-o3d/tree/master/scripts/hyperopt.py) script to
find a suitable set of parameters).
If, on the other hand, you need more fine-grained control and want to make use of the high-level abstraction provided by
the wrapping, just import the project as you would with any other Python package.

### 1. As a Python package import
This is a simple example to showcase basic functionality. Have a look at the [`tutorial.ipynb`] Jupyter notebook for
a more in depth treatment.
```python
# Import package functionality
from easy_o3d import utils
from easy_o3d.registration import RANSAC, IterativeClosestPoint

# Load source and target data
source_path = "./tests/tests/test_data/suzanne.ply"
source = utils.eval_data(data=source_path, number_of_points=10000)

target_path = "./tests/test_data/suzanne_on_chair.ply"
target = utils.eval_data(data=target_path, number_of_points=100000)

# Prepare data
source = utils.process_point_cloud(point_cloud=source,
                                   downsample=utils.DownsampleTypes.VOXEL,
                                   downsample_factor=0.01)

_, source_feature = utils.process_point_cloud(point_cloud=source,
                                              compute_feature=True,
                                              search_param_knn=100,
                                              search_param_radius=0.05)

target = utils.process_point_cloud(point_cloud=target,
                                   downsample=utils.DownsampleTypes.VOXEL,
                                   downsample_factor=0.01)

_, target_feature = utils.process_point_cloud(point_cloud=target,
                                              compute_feature=True,
                                              search_param_knn=100,
                                              search_param_radius=0.05)

# Run initializer
ransac = RANSAC()
ransac_result = ransac.run(source=source,
                           target=target,
                           source_feature=source_feature,
                           target_feature=target_feature)

# Run refiner on initializer result and visualize result
icp = IterativeClosestPoint()
icp_result = icp.run(source=source,
                     target=target,
                     init=ransac_result.transformation,
                     draw=True,
                     overwrite_colors=True)

# Load ground truth pose data and evaluate result
gt_path = "./tests/test_data/ground_truth_pose.json"
gt_transformation = utils.get_ground_truth_pose_from_file(path_to_ground_truth_json=gt_path)
error = utils.get_transformation_error(transformation_estimate=icp_result.transformation,
                                       transformation_ground_truth=gt_transformation)
```

### 2. Stand-alone using the provided scripts
The [`registration.py`](https://github.com/hummat/easy-o3d/tree/master/scripts/registration.py) script takes
a [`registration.ini`](https://github.com/hummat/easy-o3d/tree/master/scripts/registration.ini) file as input in which
paths to source and target data as well as all registration hyperparameters are specified.
```commandline
python scripts/registration.py -c scripts/registration.ini
```

To find a suitable set of hyperparameters, the
[`hyperopt.py`](https://github.com/hummat/easy-o3d/tree/master/scripts/hyperopt.py) script can be used which takes a
[`hyperopt.ini`](https://github.com/hummat/easy-o3d/tree/master/scripts/hyperopt.ini) file as input, resembling the
[`registration.ini`](https://github.com/hummat/easy-o3d/tree/master/scripts/registration.ini) file, but specifies
ranges of values to search over, and produces a
[`registration.ini`](https://github.com/hummat/easy-o3d/tree/master/scripts/registration.ini) as output with the optimal
parameter values found during the hyperparameter search. This can in turn directly be used as input to the
[`registration.py`](https://github.com/hummat/easy-o3d/tree/master/scripts/registration.py) script.
```commandline
python scripts/hyperopt.py -c scripts/hyperopt.ini
```

## Credits
First and foremost, a huge thanks for the creators and contributors of the awesome [Open3D](http://www.open3d.org) package!
The example data used throughout this project was created with [Blender](https://www.blender.org/) and
[BlenderProc](https://github.com/DLR-RM/BlenderProc). The former being an awesome open source 3D creation software and
the latter being an equally awesome open source synthetic training data generator based on Blender targeted at machine
learning applications.