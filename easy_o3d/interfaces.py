"""Interfaces and base classes.

Classes:
    ICPTypes: Supported ICP registration types.
    MetricTypes: Supported registration quality metric types.
    MyRegistrationResult: Helper class mimicking Open3D's `RegistrationResult` but mutable and with added runtime.
    RegistrationInterface: Interface for all registration classes.
"""
import copy
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Union, Dict, Tuple, List
import time
from enum import Flag, auto
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import tqdm

import numpy as np
import open3d as o3d

from .utils import (InputTypes, DownsampleTypes, SearchParamTypes, eval_data, draw_geometries,
                    process_point_cloud, TransformationTypes)

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


class MetricTypes(Flag):
    """Supported registration quality metric types."""
    RMSE = auto()
    FITNESS = auto()
    SUM = auto()
    BOTH = auto()


class MyRegistrationResult:
    """Helper class mimicking Open3D's `RegistrationResult` but mutable and with added runtime."""

    def __init__(self,
                 correspondence_set: np.ndarray,
                 fitness: float,
                 inlier_rmse: float,
                 transformation: np.ndarray,
                 runtime: float):
        self.correspondence_set = correspondence_set
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse
        self.transformation = transformation
        self.runtime = runtime


class RegistrationInterface(ABC):
    """Interface for registration subclasses. Handles data caching and visualization.

    Attributes:
        name: The name of the registration algorithm.
        auto_cache: Automatically cache anything passing trough this function that is not already cached.
        cache_size: Maximum number of elements allowed in cache.
        _parallel: Thread pool.
        _cached_data: A dictionary for caching data used during registration.

    Methods:
        parallel: Lazy-loading of multi-thread parallel pool as class property.
        _eval_data(data_key_or_value): Adds caching to `utils.eval_data`.
        _eval_data_parallel(data_keys_or_values): Runs _eval_data in parallel threads.
        _compute_dist(point_cloud): Returns maximum correspondence distance for registration based on `point_cloud`
                                    size.
        add_to_cache(data, replace): Adds `data.values()` to cache, replacing data with existing dict keys if `replace`
                                     is set to `True`.
        replace_in_cache(data): Replaces `data.values()` for existing keys.
        get_cache_value(data_key): Returns value for `data_key` from cache.
        get_cache_key(data_value): Returns key for `cached_value` from cache.
        is_in_cache(data_key_or_value): Checks if key is in cached data.
        draw_registration_result(source, target, pose, ...): Visualizes the registration result of `source` being
                                                             aligned with `target` using `pose`.
        run(source, target, ...): Runs the registration algorithm of the derived class.
        run_n_times(source, target, n_times, ...): Runs the registration algorithm of the derived class N times.
        run_multi_scale(source, target, ...): Runs the instantiated registration method at multiple scales.
        run_many(source_list, target_list, ...): Convenience function to register multiple sources and targets.
                                                 Wraps `run`, `run_n_times`, `run_multi_scale`.
    """

    def __init__(self,
                 name: str,
                 data_to_cache: Union[Dict[Any, InputTypes], None] = None,
                 auto_cache: bool = True,
                 cache_size: int = 100) -> None:
        """
        Args:
            name: The name of the registration algorithm.
            data_to_cache: The data to be cached.
            auto_cache: Automatically cache anything passing trough this function that is not already cached.
            cache_size: Maximum number of elements allowed in cache.
        """
        self.name = name
        self.auto_cache = auto_cache
        self.cache_size = cache_size
        self._parallel = None
        self._cached_data = dict()
        if data_to_cache is not None:
            self.add_to_cache(data=data_to_cache)

    @property
    def parallel(self):
        if self._parallel is None:
            self._parallel = Parallel(n_jobs=cpu_count(), prefer="threads")
        return self._parallel

    def _eval_data(self,
                   data_key_or_value: InputTypes,
                   **kwargs: Any) -> PointCloud:
        """Returns cached data if possible. Processes input to return point cloud otherwise.

        Args:
            data_key_or_value: Either data to be evaluated or key to cached data.

        Returns:
            The cached or loaded/processed point cloud data.
        """
        if isinstance(data_key_or_value, PointCloud):
            return data_key_or_value
        elif data_key_or_value in self._cached_data:
            return self.get_cache_value(data_key_or_value)
        elif self.auto_cache and len(self._cached_data) < self.cache_size:
            self.add_to_cache(data={data_key_or_value: data_key_or_value}, **kwargs)
            return self.get_cache_value(data_key_or_value)
        else:
            return eval_data(data=data_key_or_value, **kwargs)

    def _eval_data_parallel(self, data_keys_or_values: List[InputTypes], **kwargs: Any) -> List[PointCloud]:
        """Runs _eval_data in parallel threads.

        Args:
            data_keys_or_values: List of data to be evaluated or keys to cached data.

        Returns:
            List of cached or loaded/processed point cloud data.
        """
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
                     replace: bool = True,
                     **kwargs: Any) -> None:
        """Adds `data.values()` to cache, replacing data with existing dict keys if `replace` is set to `True`.

        Args:
            data: The data to be cached.
            replace: Overwrite existing data in cache.
        """
        for key, value in data.items():
            _value = eval_data(data=value, **kwargs)
            if not (self.is_in_cache(key) or self.is_in_cache(_value)):
                if len(self._cached_data) >= self.cache_size:
                    first_key = list(self._cached_data.keys())[0]
                    logger.warning(f"Cache is full. Removing data with key {first_key} and value"
                                   f"{self._cached_data[first_key]}.")
                    self._cached_data.pop(first_key)
                self._cached_data[key] = _value
            elif replace:
                self.replace_in_cache(data={key: _value})

    def replace_in_cache(self, data: Dict[Any, InputTypes], **kwargs: Any) -> None:
        """Replaces `data.values()` for existing keys or `data.keys()` for existing values.

        Args:
            data: The data to be replaced.
        """
        for key, value in data.items():
            _value = eval_data(data=value, **kwargs)
            if self.is_in_cache(key):
                logger.debug(f"Replacing data with key {key} in cache.")
                self._cached_data[key] = _value
            elif self.is_in_cache(_value):
                current_key = self.get_cache_key(_value)
                logger.debug(f"Replacing key {current_key} for data with value {value} with key {key}.")
                self._cached_data[key] = _value
                self._cached_data.pop(current_key)

    def get_cache_value(self, data_key: Any) -> PointCloud:
        """Returns value for `data_key` from cache.

        Args:
            data_key: Key storing value in cache.

        Returns:
            The cached point cloud.
        """
        return self._cached_data[data_key]

    def get_cache_key(self, cached_value: PointCloud) -> Any:
        """Returns key for `cached_value` from cache.

        Args:
            cached_value: Point cloud stored in cache.

        Returns:
            Key to stored point cloud in cache.
        """
        return list(self._cached_data.keys())[list(self._cached_data.values()).index(cached_value)]

    def is_in_cache(self, data_key_or_value: InputTypes) -> bool:
        """Checks if key is in cached data.

        Args:
            data_key_or_value: The key or value to be checked.

        Returns:
            `True` if key or value is in cached data, `False` otherwise.
        """
        return data_key_or_value in self._cached_data or data_key_or_value in self._cached_data.values()

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

        draw_geometries(geometries=to_draw, window_name=f"{self.name} Registration Result", **kwargs)

    @abstractmethod
    def run(self,
            source: InputTypes,
            target: InputTypes,
            init: TransformationTypes = np.eye(4),
            **kwargs: Any) -> MyRegistrationResult:
        """Runs the registration algorithm of the derived class.

        Args:
            source: The source data.
            target: The target data.
            init: The initial pose of `source`. Can be translation, rotation, transformation or "center", in which case
                  `source` is translated to `target` center.

        Raises:
            NotImplementedError: A derived class should implement this method.

        Returns:
            The registration result containing relative fitness (`fitness`) and RMSE (`relative_rmse`) as well as the
            correspondence set between `source` and `target` (`correspondence_set`) and transformation
            (`transformation`) between `source` and `target` and runtime (`runtime`).
        """
        raise NotImplementedError("A derived class should implement this method.")

    def run_n_times(self,
                    source: InputTypes,
                    target: InputTypes,
                    init: TransformationTypes = np.eye(4),
                    n_times: int = 3,
                    threshold: float = 0.0,
                    metric: MetricTypes = MetricTypes.BOTH,
                    multi_scale: bool = False,
                    **kwargs: Any) -> MyRegistrationResult:
        """Runs the registration algorithm of the derived class N times.

        Args:
            source: The source data.
            target: The target data.
            init: The initial pose of `source`. Can be translation, rotation or transformation.
            n_times: How often to run before returning best result. If -1, runs until below/above metric threshold.
            threshold: Runs until below/above metric threshold if `n_times` is -1.
            metric: Which metric to use to decide which result is best.
            multi_scale: Use multi-scale registration instead of single scale.

        Returns:
            The best registration result from N runs.
        """
        best_result = MyRegistrationResult(correspondence_set=np.array([]),
                                           inlier_rmse=np.infty,
                                           fitness=0.0,
                                           transformation=np.eye(4),
                                           runtime=0.0)

        _func = self.run_multi_scale if multi_scale else self.run

        def _eval_result(_result, _best_result):
            _metric = 0
            if metric == MetricTypes.RMSE:
                if _result.inlier_rmse < _best_result.inlier_rmse and len(_result.correspondence_set) > 0:
                    _best_result = _result
            elif metric == MetricTypes.FITNESS:
                if _result.fitness > _best_result.fitness and len(_result.correspondence_set) > 0:
                    _best_result = _result
            elif metric == MetricTypes.SUM:
                if _result.fitness - _result.inlier_rmse > _best_result.fitness - _result.inlier_rmse and len(_result.correspondence_set) > 0:
                    _best_result = _result
            elif metric == MetricTypes.BOTH:
                if _result.fitness > _best_result.fitness and _result.inlier_rmse < best_result.inlier_rmse and len(_result.correspondence_set) > 0:
                    _best_result = _result
            else:
                raise ValueError(f"{metric} is not a valid `MetricsType` type.")
            return _best_result

        if n_times != -1:
            for n in range(n_times):
                result = _func(source=source, target=target, init=init, **kwargs)
                best_result = _eval_result(result, best_result)
        else:
            if metric == MetricTypes.RMSE:
                while best_result.inlier_rmse > threshold:
                    result = _func(source=source, target=target, init=init, **kwargs)
                    best_result = _eval_result(result, best_result)
            elif metric == MetricTypes.FITNESS:
                while best_result.fitness < threshold:
                    result = _func(source=source, target=target, init=init, **kwargs)
                    best_result = _eval_result(result, best_result)
            elif metric == MetricTypes.SUM:
                while best_result.fitness - best_result.inlier_rmse < threshold:
                    result = _func(source=source, target=target, init=init, **kwargs)
                    best_result = _eval_result(result, best_result)
            elif metric == MetricTypes.BOTH:
                raise ValueError(f"Can't use threshold-based return with metric `BOTH`.")
            else:
                raise ValueError(f"{metric} is not a valid `MetricsType` type.")
        return best_result

    def run_multi_scale(self,
                        source: InputTypes,
                        target: InputTypes,
                        init: TransformationTypes = np.eye(4),
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
        start = time.time()
        if target_scales is None:
            target_scales = source_scales
        assert len(source_scales) == len(iterations) == len(target_scales),\
            "Need to provide same number of 'source_scales', 'target_scales' and 'iterations'."

        _source = self._eval_data(data_key_or_value=source, **kwargs)
        _target = self._eval_data(data_key_or_value=target, **kwargs)

        current_result = None
        current_transformation = init
        for i, (source_scale, target_scale, iteration) in enumerate(zip(source_scales, target_scales, iterations)):
            source_radius = radius_multiplier[0] * kwargs.get("search_param_radius", source_scale)
            target_radius = radius_multiplier[0] * kwargs.get("search_param_radius", target_scale)
            logger.debug(f"Iteration {i + 1}/{len(iterations)} with scales={source_scale, target_scale},"
                         f"iterations={iteration}, radii={source_radius, target_radius}")

            source_down = process_point_cloud(point_cloud=_source,
                                              downsample=kwargs.get("downsample", DownsampleTypes.VOXEL),
                                              downsample_factor=source_scale,
                                              estimate_normals=False if "POINT_TO_POINT" in self.name else True,
                                              recalculate_normals=kwargs.get("recalculate_normals", False),
                                              orient_normals=kwargs.get("orient_normals", None),
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_radius=source_radius,
                                              search_param_knn=kwargs.get("search_param_knn", 30))
            target_down = process_point_cloud(point_cloud=_target,
                                              downsample=kwargs.get("downsample", DownsampleTypes.VOXEL),
                                              downsample_factor=target_scale,
                                              estimate_normals=False if "POINT_TO_POINT" in self.name else True,
                                              recalculate_normals=kwargs.get("recalculate_normals", False),
                                              orient_normals=kwargs.get("orient_normals", None),
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_radius=target_radius,
                                              search_param_knn=kwargs.get("search_param_knn", 30))
            if "ICP" in self.name:
                current_result = self.run(source=source_down,
                                          target=target_down,
                                          max_correspondence_distance=source_radius,
                                          init=current_transformation,
                                          max_iteration=iteration,
                                          **kwargs)
            else:
                source_radius = radius_multiplier[1] * kwargs.get("search_param_radius", source_scale)
                target_radius = radius_multiplier[1] * kwargs.get("search_param_radius", target_scale)

                source_down, source_feature = process_point_cloud(point_cloud=source_down,
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
                                          init=current_transformation,
                                          max_correspondence_distance=source_radius,
                                          source_feature=source_feature,
                                          target_feature=target_feature,
                                          max_iteration=iteration,
                                          **kwargs)
            current_transformation = current_result.transformation
            current_result.runtime = time.time() - start
        return current_result

    def run_many(self,
                 source_list: List[InputTypes],
                 target_list: List[InputTypes],
                 init_list: Union[TransformationTypes, None] = None,
                 one_vs_one: bool = False,
                 n_times: int = 1,
                 multi_scale: bool = False,
                 progress: bool = True,
                 **kwargs: Any) -> List[MyRegistrationResult]:
        """Convenience function to register multiple sources and targets. Wraps `run`, `run_n_times`, `run_multi_scale`.

        Args:
            source_list: A list of sources.
            target_list: A list of targets.
            init_list: The initial poses of sources in `source_list`. Can be translation, rotation or transformation.
            one_vs_one: Register one source to one target. Otherwise, each source is registered to each target.
            n_times: How often to run before returning best result. Further documentation at `run_n_times`.
            multi_scale: Use multi-scale registration instead of single scale.
            progress: Print progress bar.

        Returns:
            A list of registration results between
            source_0 <-> target_0, source_1 <-> target_0, ... source_N <-> target_0, source_0 <-> target_1, ...
            If `one_vs_one`, the order is source_0 <-> target_0, source_1 <-> target_1, ...
        """
        start = time.time()

        is_list = True
        if init_list is None:
            init_list = np.eye(4)
            is_list = False
        elif isinstance(init_list, str):
            is_list = False
        # A single rotation, translation or transformation in natural or homogeneous coordinates
        if np.asarray(init_list).size in [3, 4, 6, 9, 12, 16] and all(isinstance(value, (float, int)) for value in init_list):
            is_list = False
            # A single transformation with Euler, quaternion or matrix rotation and translation in natural or
            # homogeneous coordinates
        elif len(init_list) == 2 and np.asarray(init_list[0]).size in [3, 4, 9] and np.asarray(init_list[1]).size in [3, 4]:
            is_list = False

        results = list()
        if one_vs_one and len(source_list) == len(target_list):
            if is_list:
                assert len(init_list) == len(source_list), f"'init_list' and 'source_list' must have equal length."
            for source, target in tqdm.tqdm(zip(source_list, target_list),
                                            desc=self.name,
                                            file=sys.stdout,
                                            disable=not progress):
                if n_times == 1:
                    if multi_scale:
                        results.append(self.run_multi_scale(source=source,
                                                            target=target,
                                                            init=init_list.pop(0) if is_list else init_list,
                                                            **kwargs))
                    else:
                        results.append(self.run(source=source,
                                                target=target,
                                                init=init_list.pop(0) if is_list else init_list,
                                                **kwargs))
                else:
                    results.append(self.run_n_times(source=source,
                                                    target=target,
                                                    init=init_list.pop(0) if is_list else init_list,
                                                    n_times=n_times,
                                                    multi_scale=multi_scale,
                                                    **kwargs))
        else:
            if one_vs_one:
                logger.warning(f"Source and target list have unequal length which is required for `one_vs_one`.")
            if is_list:
                assert len(init_list) == len(source_list) * len(target_list),\
                    f"'len(init_list)' must equal 'len(source_list) * len(target_list)."
            progress = tqdm.tqdm(range(len(source_list) * len(target_list)),
                                 file=sys.stdout,
                                 desc=self.name,
                                 disable=not progress)
            for target in target_list:
                for source in source_list:
                    if n_times == 1:
                        if multi_scale:
                            results.append(self.run_multi_scale(source=source,
                                                                target=target,
                                                                init=init_list.pop(0) if is_list else init_list,
                                                                **kwargs))
                        else:
                            results.append(self.run(source=source,
                                                    target=target,
                                                    init=init_list.pop(0) if is_list else init_list,
                                                    **kwargs))
                    else:
                        results.append(self.run_n_times(source=source,
                                                        target=target,
                                                        init=init_list.pop(0) if is_list else init_list,
                                                        n_times=n_times,
                                                        multi_scale=multi_scale,
                                                        **kwargs))
                    progress.update()
            progress.close()
        logger.debug(f"`run_many` took {time.time() - start} seconds.")
        return results
