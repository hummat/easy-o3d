# Easy Open3D
This is an easy-to-use wrapper around, as well as utility functions and scripts for, some of
[Open3D](http://www.open3d.org/) 's registration functionality.

_Head over to the repository's [**GitHub** pages site](https://hummat.github.io/easy-o3d) for a more interactive
version of this README!_

## What is registration?
In 3D data analysis, the term _registration_ usually refers to the process of aligning two partly overlapping point
clouds such that the result is a merged version of the input. It is frequently used to obtain a 3D scan of a large scene
by stitching together several smaller parts.
In robotics however, we can use the same algorithms to register a 3D model of an object we want to manipulate to the
part of the 3D scan (obtained through depth sensors) of the scene the robot operates in to find its pose in this scene.
The pose of an object consists of its rotation and translation in some coordinate frame, typically the
robots head frame, camera frame or world frame.

### A simple yet slightly contrived example
TODO

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
also install the optional dependency `scikit-optimize` with:
```commandline
pip install /path/to/easy-o3d/repository/clone[hyper]
```
(or manually with `pip install scikit-optimize`).

Alternatively, you can install the [required packages](https://github.com/hummat/easy-o3d/blob/master/requirements.txt)
manually and directly import the code from your local clone of the repository:
```commandline
pip install /path/to/requirements.txt
```
The required packages and their version used during development are:
1. `numpy == 1.20.2`
2. `open3d == 0.12.0`
3. (`scikit-optimize == 0.8.1`)
```commandline
pip install numpy==1.20.2 open3d==0.12.0 scikit-optimize==0.8.1
```

## Usage
Fundamentally, there are two ways to use this project. You can either import it into your existing code base as you
would do with any other Python package. If you only want to find the pose of some objects in some scenes, simply throw
them at the `run_registration.py` script. If, on the other hand, you need more fine-grained control and want to make
use of the high-level abstraction provided by the wrapping, just import the project as you would with any other Python
package.

### 1. As a Python package import
### 2. Stand-alone using the provided scripts
## Credits