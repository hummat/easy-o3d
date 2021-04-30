"""Interfaces and base classes.

Classes:
    ICPTypes: Supported ICP registration types.
    MyRegistrationResult: Helper class mimicking Open3D's `RegistrationResult` but mutable.
    RegistrationInterface: Interface for all registration classes.
"""

import copy
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Union, Dict, Tuple, List
import time
from enum import Flag, auto
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import numpy as np
import open3d as o3d

from .utils import (InputTypes, DownsampleTypes, SearchParamTypes, OrientationTypes, eval_data, draw_geometries,
                    process_point_cloud)

PointToPoint = o3d.pipelines.registration.TransformationEstimationPointToPoint
PointToPlane = o3d.pipelines.registration.TransformationEstimationPointToPlane
ColoredICP = o3d.pipelines.registration.TransformationEstimationForColoredICP
RegistrationResult = o3d.pipelines.registration.RegistrationResult
TriangleMesh = o3d.geometry.TriangleMesh
PointCloud = o3d.geometry.PointCloud
Feature = o3d.pipelines.registration.Feature

logger = logging.getLogger(__name__)


class ICPTypes(Flag):
    """Supported ICP registration types."""
    POINT = auto()
    PLANE = auto()
    COLOR = auto()


class MyRegistrationResult:
    """Helper class mimicking Open3D's `RegistrationResult` but mutable."""

    def __init__(self,
                 correspondence_set: np.ndarray,
                 fitness: float,
                 inlier_rmse: float,
                 transformation: np.ndarray):
        self.correspondence_set = correspondence_set
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse
        self.transformation = transformation


class RegistrationInterface(ABC):
    """Interface for registration subclasses. Handles data caching and visualization.

    Attributes:
        _name: The name of the registration algorithm.
        _cached_data: A dictionary for caching data used during registration.

    Methods:
        _eval_data(data_key_or_value): Adds caching to `utils.eval_data`.
        _compute_dist(point_cloud): Returns maximum correspondence distance for registration based on `point_cloud`
                                    size.
        add_to_cache(data, replace): Adds `data.values()` to cache, replacing data with existing dict keys if `replace`
                                     is set to `True`.
        replace_in_cache(data): Replaces `data.values()` for existing keys.
        get_from_cache(data_key): Returns value for `data_key` from cache.
        draw_registration_result(source, target, pose, ...): Visualizes the registration result of `source` being
                                                             aligned with `target` using `pose`.
        run(source, target): Runs the registration algorithm of the derived class.
    """

    def __init__(self, name: str, data_to_cache: Union[Dict[Any, InputTypes], None] = None) -> None:
        """
        Args:
            name: The name of the registration algorithm.
            data_to_cache: The data to be cached.
        """
        self._name = name
        self._parallel = None
        self._cached_data = dict()
        if data_to_cache is not None:
            self.add_to_cache(data=data_to_cache)

    @property
    def parallel(self, prefer="threads"):
        if self._parallel is None:
            self._parallel = Parallel(n_jobs=cpu_count(), prefer=prefer)
        return self._parallel

    def _eval_data(self, data_key_or_value: InputTypes, **kwargs: Any) -> PointCloud:
        """Returns cached data if possible. Processes input to return point cloud otherwise.

        Args:
            data_key_or_value: data_key_or_value

        Returns:
            The cached or loaded/processed point cloud data.
        """
        try:
            if data_key_or_value in self._cached_data:
                logger.debug(f"Using cached data {data_key_or_value}.")
                return self._cached_data[data_key_or_value]
        except TypeError:
            _hash = hashlib.md5(data_key_or_value)
            if _hash in self._cached_data:
                return self._cached_data[_hash]
        logger.debug(f"Couldn't find data {data_key_or_value} in cache. Re-evaluating.")
        return eval_data(data=data_key_or_value,
                         number_of_points=kwargs.get("number_of_points"),
                         camera_intrinsic=kwargs.get("camera_intrinsic"))

    def _eval_data_parallel(self, data_keys_or_values: List[InputTypes], **kwargs: Any) -> List[PointCloud]:
        return self.parallel(delayed(self._eval_data)(data_key_or_value=d, **kwargs) for d in data_keys_or_values)
        
    @staticmethod
    def _compute_dist(point_cloud: PointCloud) -> float:
        """Returns maximum correspondence distance for registration based on `point_cloud` size.

        Args:
            point_cloud: The point cloud used to estimate the maximum correspondence distance.

        Returns:
            The maximum correspondence distance based on `point_cloud` size.
        """
        distance = (np.asarray(point_cloud.get_max_bound()) - np.asarray(point_cloud.get_min_bound())).max()
        logger.debug(f"Using {distance} as maximum correspondence distance.")
        return distance

    def add_to_cache(self,
                     data: Dict[Any, InputTypes],
                     replace: bool = False,
                     **kwargs: Any) -> None:
        """Adds `data.values()` to cache, replacing data with existing dict keys if `replace` is set to `True`.

        Args:
            data: The data to be cached.
            replace: Overwrite existing data in cache.
        """
        if len(data) > 10:  # Magic number: Threshold above which multi-threaded loading might start to make sense
            parallel = Parallel(n_jobs=min(cpu_count(), len(data)), prefer="threads")
            values = parallel(delayed(eval_data)(data=value,
                                                 number_of_points=kwargs.get("number_of_points"),
                                                 camera_intrinsic=kwargs.get("camera_intrinsic")) for value in data.values())
            _data = dict(zip(list(data.keys()), values))
        else:
            _data = data
        for key, value in _data.items():
            try:
                if key not in self._cached_data or replace:
                    logger.debug(f"Adding data with key {key} to cache.")
                    self._cached_data[key] = eval_data(data=value,
                                                       number_of_points=kwargs.get("number_of_points"),
                                                       camera_intrinsic=kwargs.get("camera_intrinsic"))
            except TypeError:
                logger.warning(f"Couldn't add data with unhashable key {key}.")

    def replace_in_cache(self,
                         data: Dict[Any, InputTypes],
                         **kwargs: Any) -> None:
        """Replaces `data.values()` for existing keys.

        Args:
            data: The data to be replaced.
        """
        for key, value in data.items():
            try:
                if key in self._cached_data:
                    logger.debug(f"Replacing data with key {key} in cache.")
                    self._cached_data[key] = eval_data(data=value,
                                                       number_of_points=kwargs.get("number_of_points"),
                                                       camera_intrinsic=kwargs.get("camera_intrinsic"))
            except TypeError:
                logger.warning(f"Couldn't replace data with unhashable key {key}.")

    def get_from_cache(self, data_key: InputTypes) -> PointCloud:
        """Returns value for `data_key` from cache.

        Args:
            data_key: Key storing `data.value` in cache.

        Returns:
            The cached point cloud data.
        """
        return self._cached_data[data_key]

    def is_in_cache(self, data_key: InputTypes) -> bool:
        """Checks if key is in cached data.

        Args:
            data_key: The key to be checked.

        Returns:
            `True` if key is in cached data, `False` otherwise.
        """
        try:
            return data_key in self._cached_data
        except TypeError:
            return False

    def draw_registration_result(self,
                                 source: InputTypes,
                                 target: InputTypes,
                                 pose: Union[np.ndarray, list] = np.eye(4),
                                 draw_coordinate_frames: bool = True,
                                 draw_bounding_boxes: bool = False,
                                 overwrite_colors: bool = False,
                                 **kwargs: Any) -> None:
        """Visualizes the registration result of `source` being aligned with `target` using `pose`.

        Args:
            source: The source data.
            target: The target data.
            pose: The 4x4 transformation matrix between `source` and `target`.
            draw_coordinate_frames: Draws coordinate frames for `source` and `target`.
            draw_bounding_boxes: Draws axis-aligned bounding boxes for `source` and `target`.
            overwrite_colors: Overwrites `source` and `target` colors for clearer visualization.
        """
        _source = copy.deepcopy(self._eval_data(data_key_or_value=source, **kwargs))
        _target = copy.deepcopy(self._eval_data(data_key_or_value=target, **kwargs))
        _pose = np.asarray(pose).reshape(4, 4)

        if not _source.has_colors() or overwrite_colors:
            _source.paint_uniform_color([0.8, 0, 0])
        if not _target.has_colors() or overwrite_colors:
            _target.paint_uniform_color([0.8, 0.8, 0.8])

        _source.transform(_pose)

        to_draw = [_source, _target]
        if draw_coordinate_frames:
            size = 0.5 * (np.asarray(_source.get_max_bound()) - np.asarray(_source.get_min_bound())).max()
            to_draw.append(TriangleMesh().create_coordinate_frame(size=2 * size))
            to_draw.append(TriangleMesh().create_coordinate_frame(size=size).transform(_pose))

        if draw_bounding_boxes:
            to_draw.append(_source.get_axis_aligned_bounding_box())
            to_draw.append(_target.get_axis_aligned_bounding_box())

        draw_geometries(geometries=to_draw, window_name=f"{self._name} Registration Result", **kwargs)

    @staticmethod
    def _eval_init(init: Union[np.ndarray, list]) -> np.ndarray:
        """Evaluates initial pose data to parse translation, rotation and transformation.

        Args:
            init: The initial pose.

        Returns:
            The evaluated initial pose data as a 4x4 transformation matrix.
        """
        _init = np.eye(4)
        init = np.asarray(init)
        if init.size in [3, 4]:
            _init[:3, 3] = init.ravel()[:3]
        elif init.size == 9:
            _init[:3, :3] = init.reshape(3, 3)
        elif init.size == 16:
            _init = init.reshape(4, 4)
        else:
            raise ValueError("`init` needs to be a valid translation, rotation or transformation in natural or"
                             "homogeneous coordinates, i.e. of size 3, 4, 9 or 16.")
        return _init

    @abstractmethod
    def run(self,
            source: InputTypes,
            target: InputTypes,
            **kwargs: Any) -> RegistrationResult:
        """Runs the registration algorithm of the derived class.

        Args:
            source: The source data.
            target: The target data.

        Raises:
            NotImplementedError: A derived class should implement this method.

        Returns:
            The registration result containing relative fitness (`fitness`) and RMSE (`relative_rmse`) as well as the
            correspondence set between `source` and `target` (`correspondence_set`) and transformation
            (`transformation`) between `source` and `target`.
        """
        raise NotImplementedError("A derived class should implement this method.")

    def run_multi_scale(self,
                        source: InputTypes,
                        target: InputTypes,
                        init: [np.ndarray, list] = np.eye(4),
                        source_scales: Union[Tuple[float], List[float]] = (0.04, 0.02, 0.01),
                        target_scales: Union[Tuple[float], List[float], None] = None,
                        iterations: Union[Tuple[int], List[int]] = (50, 30, 14),
                        radius_multiplier: Union[Tuple[int], List[int]] = (2, 5),
                        **kwargs: Any) -> MyRegistrationResult:
        """Runs the instantiated registration method at multiple scales.

        To increase efficiency and effectiveness, the instantiated registration method is run multiple times with
        decreasing downsampling factor, correspondence distance and number of iterations to obtain tighter alignments
        without major increase in computation time.

        Args:
            source: The source data.
            target: The target data.
            init: The initial pose of `source`. Can be translation, rotation or transformation.
            source_scales: The downsampling scales for the source data.
            target_scales: The downsampling scales for the target data. Same as `source_scales` if not provided.
            iterations: The number of iterations to run the algorithm at each downsampling stage.
            radius_multiplier: The current scale is multiplied by these to obtain the normal and FPFH feature radius.

        Returns:
            The registration result containing relative fitness (`fitness`) and RMSE (`relative_rmse`) as well as the
            correspondence set between `source` and `target` (`correspondence_set`) and transformation
            (`transformation`) between `source` and `target`.
        """
        if target_scales is None:
            target_scales = source_scales
        assert len(source_scales) == len(iterations) == len(target_scales), \
            "Need to provide the same number of scales and iterations."

        _source = self._eval_data(data_key_or_value=source, **kwargs)
        _target = self._eval_data(data_key_or_value=target, **kwargs)

        current_result = None
        current_transformation = self._eval_init(init)
        for i, (source_scale, target_scale, iteration) in enumerate(zip(source_scales, target_scales, iterations)):
            source_radius = radius_multiplier[0] * kwargs.get("search_param_radius", source_scale)
            target_radius = radius_multiplier[0] * kwargs.get("search_param_radius", target_scale)
            logger.debug(f"Iteration {i + 1}/{len(iterations)} with scales={source_scale, target_scale},"
                         f"iterations={iteration}, radii={source_radius, target_radius}")

            source_down = process_point_cloud(point_cloud=_source,
                                              downsample=kwargs.get("downsample", DownsampleTypes.VOXEL),
                                              downsample_factor=source_scale,
                                              estimate_normals=False if "POINT_TO_POINT" in self._name else True,
                                              recalculate_normals=kwargs.get("recalculate_normals", False),
                                              orient_normals=kwargs.get("orient_normals", OrientationTypes.NONE),
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_radius=source_radius,
                                              search_param_knn=kwargs.get("search_param_knn", 30))
            target_down = process_point_cloud(point_cloud=_target,
                                              downsample=kwargs.get("downsample", DownsampleTypes.VOXEL),
                                              downsample_factor=target_scale,
                                              estimate_normals=False if "POINT_TO_POINT" in self._name else True,
                                              recalculate_normals=kwargs.get("recalculate_normals", False),
                                              orient_normals=kwargs.get("orient_normals", OrientationTypes.NONE),
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_radius=target_radius,
                                              search_param_knn=kwargs.get("search_param_knn", 30))
            if "ICP" in self._name:
                current_result = self.run(source=source_down,
                                          target=target_down,
                                          max_correspondence_distance=source_radius,
                                          init=current_transformation,
                                          max_iteration=iteration,
                                          **kwargs)
                current_transformation = current_result.transformation
            else:
                source_down = source_down.transform(current_transformation)
                source_radius = radius_multiplier[1] * kwargs.get("search_param_radius", source_scale)
                target_radius = radius_multiplier[1] * kwargs.get("search_param_radius", target_scale)

                _, source_feature = process_point_cloud(point_cloud=source_down,
                                                        compute_feature=True,
                                                        search_param=kwargs.get("search_param",
                                                                                SearchParamTypes.HYBRID),
                                                        search_param_radius=source_radius,
                                                        search_param_knn=kwargs.get("search_param_knn", 100))

                _, target_feature = process_point_cloud(point_cloud=target_down,
                                                        compute_feature=True,
                                                        search_param=kwargs.get("search_param",
                                                                                SearchParamTypes.HYBRID),
                                                        search_param_radius=target_radius,
                                                        search_param_knn=kwargs.get("search_param_knn", 100))

                current_result = self.run(source=source_down,
                                          target=target_down,
                                          max_correspondence_distance=source_radius,
                                          source_feature=source_feature,
                                          target_feature=target_feature,
                                          max_iteration=iteration,
                                          **kwargs)
                current_transformation = current_result.transformation @ current_transformation
        return MyRegistrationResult(correspondence_set=current_result.correspondence_set,
                                    fitness=current_result.fitness,
                                    inlier_rmse=current_result.inlier_rmse,
                                    transformation=current_transformation)

    def run_many(self,
                 source_list: List[InputTypes],
                 target_list: List[InputTypes],
                 init_list: Union[List[np.ndarray], List[list], None] = None,
                 one_vs_one: bool = False,
                 multi_scale: bool = False,
                 multi_thread_preload: bool = True,
                 **kwargs: Any) -> Union[List[RegistrationResult], List[MyRegistrationResult]]:
        """Convenience function to register multiple sources and targets. Takes accepts all arguments accepted by `run`.

        Args:
            source_list: A list of sources.
            target_list: A list of targets.
            init_list: The initial poses of sources in `source_list`. Can be translation, rotation or transformation.
            one_vs_one: Register one source to one target. Otherwise, each source is registered to each target.
            multi_scale: Use multi-scale registration instead of single scale.
            multi_thread_preload: If input type is string, pre-loads data from disk in parallel using multi-threading.

        Returns:
            A list of registration results between
            source_0 <-> target_0, source_1 <-> target_0, ... source_N <-> target_0, source_0 <-> target_1, ...
            If `one_vs_one`, the order is source_0 <-> target_0, source_1 <-> target_1, ...
        """
        start = time.time()
        if multi_thread_preload and any(type(x) == str for x in source_list + target_list):
            source_target_list = self._eval_data_parallel(source_list + target_list)
            source_list, target_list = source_target_list[:len(source_list)], source_target_list[len(source_list):]

        results = list()
        if one_vs_one and len(source_list) == len(target_list):
            if init_list is not None:
                assert len(init_list) == len(source_list)
            for source, target in zip(source_list, target_list):
                init = self._eval_init(init_list.pop()) if init_list is not None else np.eye(4)
                if multi_scale:
                    results.append(self.run_multi_scale(source=source,
                                                        target=target,
                                                        init=init,
                                                        **kwargs))
                else:
                    results.append(self.run(source=source,
                                            target=target,
                                            init=init,
                                            **kwargs))
        else:
            if one_vs_one:
                logger.warning(f"Source and target list have unequal length which is required for `one_vs_one`.")
            if init_list is not None:
                assert len(init_list) == len(source_list) * len(target_list),\
                    f"Need to provide one initial pose for each source/target combination to be evaluated."
            for target in target_list:
                for source in source_list:
                    init = self._eval_init(init_list.pop()) if init_list is not None else np.eye(4)
                    if multi_scale:
                        results.append(self.run_multi_scale(source=source,
                                                            target=target,
                                                            init=init,
                                                            **kwargs))
                    else:
                        results.append(self.run(source=source,
                                                target=target,
                                                init=init,
                                                **kwargs))
        logger.debug(f"`run_many` took {time.time() - start} seconds.")
        return results
