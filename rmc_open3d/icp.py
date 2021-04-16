import copy
from typing import Any, Union

import numpy as np
import open3d as o3d

from .interfaces import RegistrationInterface

PointToPoint = o3d.pipelines.registration.TransformationEstimationPointToPoint
PointToPlane = o3d.pipelines.registration.TransformationEstimationPointToPlane
ColoredICP = o3d.pipelines.registration.TransformationEstimationForColoredICP


class ICP(RegistrationInterface):
    """The Iterative Closest Point algorithm."""

    def __init__(self,
                 relative_fitness: float = 1e-6,
                 relative_rmse: float = 1e-6,
                 max_iteration: int = 30,
                 max_correspondence_distance: float = -1.0,
                 estimation_method: Union[PointToPoint, PointToPlane, ColoredICP] = PointToPoint,
                 data_to_cache: dict = None,
                 **kwargs: Any) -> None:
        """The Iterative Closest Point algorithm.

        Args:
            relative_fitness (float): relative_fitness
            relative_rmse (float): relative_rmse
            max_iteration (int): max_iteration
            max_correspondence_distance (float): max_correspondence_distance
            estimation_method (Union[PointToPoint, PointToPlane, ColoredICP]): estimation_method
            data_to_cache (dict): data_to_cache
            kwargs (Any): kwargs
        """
        super().__init__(data_to_cache=data_to_cache, **kwargs)

        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse
        self.max_iteration = max_iteration

        self.max_correspondence_distance = max_correspondence_distance

        self.algorithm = o3d.pipelines.registration.registration_icp
        self.estimation_method = estimation_method()
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=self.relative_fitness,
                                                                          relative_rmse=self.relative_rmse,
                                                                          max_iteration=self.max_iteration)

    def run(self, source: Any, target: Any, **kwargs):
        _source = copy.deepcopy(self._eval_data(data=source, **kwargs))
        _target = copy.deepcopy(self._eval_data(data=target, **kwargs))

        if any(key in kwargs for key in ["relative_fitness", "relative_rmse", "max_iteration"]):
            _criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=kwargs.get("relative_fitness", self.relative_fitness),
                relative_rmse=kwargs.get("relative_rmse", self.relative_rmse),
                max_iteration=kwargs.get("max_iteration", self.max_iteration))
        else:
            _criteria = kwargs.get("criteria", self.criteria)

        max_dist = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        # Compute based on object size
        if max_dist == -1.0:
            points = np.asarray(_source.points)
            max_dist = max(points.max(axis=0) - points.min(axis=0))

        return self.algorithm(source=_source,
                              target=_target,
                              max_correspondence_distance=max_dist,
                              init=kwargs.get("init", np.eye(4)),
                              estimation_method=kwargs.get("estimation_method", self.estimation_method),
                              criteria=_criteria)


class FastGlobalRegistration(RegistrationInterface):

    def __init__(self):
        super().__init()
