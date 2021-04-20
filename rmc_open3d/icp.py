import copy
import logging
from enum import Flag, auto
from typing import Any

import numpy as np
import open3d as o3d

from .interfaces import RegistrationInterface
from .utils import InputTypes, process_point_cloud

PointToPoint = o3d.pipelines.registration.TransformationEstimationPointToPoint
PointToPlane = o3d.pipelines.registration.TransformationEstimationPointToPlane
ColoredICP = o3d.pipelines.registration.TransformationEstimationForColoredICP
RegistrationResult = o3d.pipelines.registration.RegistrationResult


class EstimationMethodTypes(Flag):
    POINT_TO_POINT = auto()
    POINT_TO_PLANE = auto()
    COLORED = auto()


logger = logging.getLogger(__name__)


class ICP(RegistrationInterface):
    """The Iterative Closest Point algorithm."""

    def __init__(self,
                 relative_fitness: float = 1e-6,
                 relative_rmse: float = 1e-6,
                 max_iteration: int = 30,
                 max_correspondence_distance: float = -1.0,
                 estimation_method: EstimationMethodTypes = EstimationMethodTypes.POINT_TO_POINT,
                 data_to_cache: dict = None,
                 **kwargs: Any) -> None:
        """The Iterative Closest Point algorithm.

        Args:
            relative_fitness (float): relative_fitness
            relative_rmse (float): relative_rmse
            max_iteration (int): max_iteration
            max_correspondence_distance (float): max_correspondence_distance
            estimation_method (EstimationMethodTypes): estimation_method
            data_to_cache (dict): data_to_cache
            kwargs (Any): kwargs
        """
        super().__init__(data_to_cache=data_to_cache, **kwargs)

        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse
        self.max_iteration = max_iteration

        self.max_correspondence_distance = max_correspondence_distance

        self.algorithm = o3d.pipelines.registration.registration_icp
        assert isinstance(estimation_method, EstimationMethodTypes)
        if estimation_method == EstimationMethodTypes.POINT_TO_POINT:
            self.estimation_method = PointToPoint()
        elif estimation_method == EstimationMethodTypes.POINT_TO_PLANE:
            self.estimation_method = PointToPlane()
        elif estimation_method == EstimationMethodTypes.COLORED:
            self.estimation_method = ColoredICP()
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=self.relative_fitness,
                                                                          relative_rmse=self.relative_rmse,
                                                                          max_iteration=self.max_iteration)

    def run(self,
            source: InputTypes,
            target: InputTypes,
            init: np.ndarray = np.eye(4),
            crop_target_around_source: bool = False,
            crop_scale: float = 1.0,
            visualize: bool = False,
            **kwargs: Any) -> RegistrationResult:

        _source = self._eval_data(data=source, **kwargs)
        _target = self._eval_data(data=target, **kwargs)

        if any(key in kwargs for key in ["relative_fitness", "relative_rmse", "max_iteration"]):
            _criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=kwargs.get("relative_fitness", self.relative_fitness),
                relative_rmse=kwargs.get("relative_rmse", self.relative_rmse),
                max_iteration=kwargs.get("max_iteration", self.max_iteration))
        else:
            _criteria = kwargs.get("criteria", self.criteria)

        max_dist = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        # Heuristic: One 1th of object size
        if max_dist == -1.0:
            max_dist = (np.asarray(_source.get_max_bound()) - np.asarray(_source.get_min_bound())).max()
            logger.debug(f"Using {max_dist} as maximum correspondence distance.")

        if crop_target_around_source:
            bounding_box = copy.deepcopy(_source).transform(init).get_axis_aligned_bounding_box()
            bounding_box = bounding_box.scale(scale=crop_scale, center=bounding_box.get_center())
            cropped_target = copy.deepcopy(_target).crop(bounding_box=bounding_box)
            _target = cropped_target if not cropped_target.is_empty() else _target

        estimation_method = kwargs.get("estimation_method", self.estimation_method)
        if isinstance(estimation_method, (PointToPlane, ColoredICP)):
            if not _source.has_normals():
                logger.warning(
                    f"Source point cloud doesn't have normals which are needed for {type(estimation_method)}. Computing with (potentially suboptimal) default parameters."
                )
                _source = process_point_cloud(point_cloud=_source, estimate_normals=True)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={source: _source})
            if not _target.has_normals():
                logger.warning(
                    f"Target point cloud doesn't have normals which are needed for {type(estimation_method)}. Computing with (potentially suboptimal) default parameters."
                )
                _target = process_point_cloud(point_cloud=_target, estimate_normals=True)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={target: _target})

        result = self.algorithm(source=_source,
                                target=_target,
                                max_correspondence_distance=max_dist,
                                init=init,
                                estimation_method=estimation_method,
                                criteria=_criteria)

        if visualize:
            self.draw_registration_result(source=_source, target=_target, pose=result.transformation, **kwargs)

        return result


class FastGlobalRegistration(RegistrationInterface):

    def __init__(self):
        super().__init()
