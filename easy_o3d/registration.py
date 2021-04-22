"""Point cloud registration functionality.

Classes:
    registration.IterativeClosestPoint: ICP functionality.
    registration.FastGlobalRegistration: FGR functionality.
    registration.RANSAC: RANSAC functionality.
    registration.EstimationMethodTypes: Estimation method style flags.
    registration.CheckerTypes: RANSAC correspondence checker style flags.
"""

import copy
import logging
from enum import Flag, auto
from typing import Any, Union, Tuple

import numpy as np
import open3d as o3d

from .interfaces import RegistrationInterface
from .utils import InputTypes, process_point_cloud

PointToPoint = o3d.pipelines.registration.TransformationEstimationPointToPoint
PointToPlane = o3d.pipelines.registration.TransformationEstimationPointToPlane
ColoredICP = o3d.pipelines.registration.TransformationEstimationForColoredICP
TransformationEstimation = o3d.pipelines.registration.TransformationEstimation
ICPConvergenceCriteria = o3d.pipelines.registration.ICPConvergenceCriteria
FastGlobalRegistrationOption = o3d.pipelines.registration.FastGlobalRegistrationOption
Feature = o3d.pipelines.registration.Feature
RANSACConvergenceCriteria = o3d.pipelines.registration.RANSACConvergenceCriteria
CorrespondenceCheckerBasedOnEdgeLength = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength
CorrespondenceCheckerBasedOnDistance = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance
CorrespondenceCheckerBasedOnNormal = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal
ransac_feature = o3d.pipelines.registration.registration_ransac_based_on_feature_matching
ransac_correspondence = o3d.pipelines.registration.registration_ransac_based_on_correspondence
RegistrationResult = o3d.pipelines.registration.RegistrationResult


class EstimationMethodTypes(Flag):
    POINT_TO_POINT = auto()
    POINT_TO_PLANE = auto()
    COLORED = auto()


class CheckerTypes(Flag):
    EDGE_LENGTH = auto()
    DISTANCE = auto()
    NORMAL = auto()


logger = logging.getLogger(__name__)


def eval_estimation_method(estimation_method: Union[EstimationMethodTypes, TransformationEstimation]) -> Union[
    PointToPoint, PointToPlane, ColoredICP]:
    assert isinstance(estimation_method, (EstimationMethodTypes, TransformationEstimation))
    if isinstance(estimation_method, TransformationEstimation):
        return estimation_method
    if estimation_method == EstimationMethodTypes.POINT_TO_POINT:
        return PointToPoint(with_scaling=False)
    elif estimation_method == EstimationMethodTypes.POINT_TO_PLANE:
        return PointToPlane(kernel=o3d.pipelines.registration.TukeyLoss(k=0.01))
    elif estimation_method == EstimationMethodTypes.COLORED:
        return ColoredICP()
    else:
        raise ValueError(f"{estimation_method} is not a valid estimation method type.")


class IterativeClosestPoint(RegistrationInterface):
    """The *Iterative Closest Point* algorithm."""

    def __init__(self,
                 relative_fitness: float = 1e-6,
                 relative_rmse: float = 1e-6,
                 max_iteration: int = 30,
                 max_correspondence_distance: float = 0.004,  # 4mm
                 estimation_method: EstimationMethodTypes = EstimationMethodTypes.POINT_TO_POINT,
                 data_to_cache: dict = None,
                 **kwargs: Any) -> None:
        """
        Args:
            relative_fitness: If relative change (difference) of fitness score is lower than `relative_fitness`,
                              the iteration stops.
            relative_rmse: If relative change (difference) of inliner RMSE score is lower than `relative_rmse`,
                           the iteration stops.
            max_iteration: Maximum iteration before iteration stops.
            max_correspondence_distance: Maximum correspondence points-pair distance.
            estimation_method: The estimation method. One of `POINT_TO_POINT`, `POINT_TO_PLANE` or `COLORED`.
            data_to_cache: The data to be cached.
            kwargs: Optional additional keyword arguments used downstream.
        """
        super().__init__(data_to_cache=data_to_cache, **kwargs)

        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse
        self.max_iteration = max_iteration
        self.max_correspondence_distance = max_correspondence_distance

        self.algorithm = o3d.pipelines.registration.registration_icp
        self.estimation_method = eval_estimation_method(estimation_method=estimation_method)
        if isinstance(self.estimation_method, ColoredICP):
            self.algorithm = o3d.pipelines.registration.registration_colored_icp

        self.criteria = ICPConvergenceCriteria(relative_fitness=self.relative_fitness,
                                               relative_rmse=self.relative_rmse,
                                               max_iteration=self.max_iteration)

    def run(self,
            source: InputTypes,
            target: InputTypes,
            init: np.ndarray = np.eye(4),
            crop_target_around_source: bool = False,
            crop_scale: float = 1.0,
            draw: bool = False,
            **kwargs: Any) -> RegistrationResult:
        """Runs the *Iterative Closest Point* algorithm.

        The goal is to find the rotation and translation, i.e. 6D pose, of the `source` object, best
        resembling its actual pose found in the `target` point cloud.

        Args:
            source (InputTypes): The source data.
            target (InputTypes): The target data.
            init (np.ndarray): The initial pose of `source`.
            crop_target_around_source (bool): Crops `target` around the bounding box of `source`. Should only
                                              be used if `init` is already quite accurate.
            crop_scale (float): The scale of the `source` bounding box used for cropping `target`.
                                Increase if `init` is inaccurate.
            draw (bool): Visualize the registration result.
            kwargs (Any): Optional additional keyword arguments. Allows to set thresholds and parameters for ICP.
                          Please refer to the class constructor documentation.

        Returns:
            RegistrationResult: The registration result containing relative fitness and RMSE
                                as well as the the transformation between `source` and `target`.
        """
        _source = self._eval_data(data_key_or_value=source, **kwargs)
        _target = self._eval_data(data_key_or_value=target, **kwargs)

        if any(key in kwargs for key in ["relative_fitness", "relative_rmse", "max_iteration"]):
            self.criteria = ICPConvergenceCriteria(
                relative_fitness=kwargs.get("relative_fitness", self.relative_fitness),
                relative_rmse=kwargs.get("relative_rmse", self.relative_rmse),
                max_iteration=kwargs.get("max_iteration", self.max_iteration))

        self.max_correspondence_distance = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        if self.max_correspondence_distance == -1.0:
            self.max_correspondence_distance = self._compute_dist(point_cloud=_source)

        if "estimation_method" in kwargs:
            self.estimation_method = eval_estimation_method(kwargs.get("estimation_method", self.estimation_method))
        if isinstance(self.estimation_method, ColoredICP):
            self.algorithm = o3d.pipelines.registration.registration_colored_icp

        if crop_target_around_source:
            bounding_box = copy.deepcopy(_source).transform(init).get_axis_aligned_bounding_box()
            bounding_box = bounding_box.scale(scale=crop_scale, center=bounding_box.get_center())
            cropped_target = copy.deepcopy(_target).crop(bounding_box=bounding_box)
            _target = cropped_target if not cropped_target.is_empty() else _target

        if isinstance(self.estimation_method, (PointToPlane, ColoredICP)):
            if not _source.has_normals():
                logger.warning(f"Source has no normals which are needed for {self.estimation_method}.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _source = process_point_cloud(point_cloud=_source,
                                              estimate_normals=True,
                                              search_param_knn=30,
                                              search_param_radius=0.02)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={source: _source})
            if not _target.has_normals():
                logger.warning(f"Target has no normals which are needed for {self.estimation_method}.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _target = process_point_cloud(point_cloud=_target,
                                              estimate_normals=True,
                                              search_param_knn=30,
                                              search_param_radius=0.02)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={target: _target})

        result = self.algorithm(source=_source,
                                target=_target,
                                max_correspondence_distance=self.max_correspondence_distance,
                                init=init,
                                estimation_method=self.estimation_method,
                                criteria=self.criteria)

        if draw:
            self.draw_registration_result(source=_source, target=_target, pose=result.transformation, **kwargs)

        return result


class FastGlobalRegistration(RegistrationInterface):
    """The *Fast Global Registration* algorithm."""

    def __init__(self,
                 max_iteration: int = 64,
                 max_correspondence_distance: float = 0.005,  # 5mm
                 data_to_cache: dict = None,
                 **kwargs: Any) -> None:
        super().__init__(data_to_cache=data_to_cache, **kwargs)

        self.max_correspondence_distance = max_correspondence_distance
        self.max_iteration = max_iteration
        self.algorithm = o3d.pipelines.registration.registration_fast_based_on_feature_matching

        self.option = FastGlobalRegistrationOption(maximum_correspondence_distance=self.max_correspondence_distance,
                                                   iteration_number=self.max_iteration)

    def run(self,
            source: InputTypes,
            target: InputTypes,
            source_feature: Feature = None,
            target_feature: Feature = None,
            draw: bool = False,
            **kwargs: Any) -> RegistrationResult:

        _source = self._eval_data(data_key_or_value=source, **kwargs)
        _target = self._eval_data(data_key_or_value=target, **kwargs)

        self.max_correspondence_distance = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        if self.max_correspondence_distance == -1.0:
            self.max_correspondence_distance = self._compute_dist(point_cloud=_source)

        if any(key in kwargs for key in ["max_iteration"]):
            self.option = FastGlobalRegistrationOption(maximum_correspondence_distance=self.max_correspondence_distance,
                                                       iteration_number=kwargs.get("max_iteration", self.max_iteration))

        if source_feature is None:
            logger.warning("Source FPFH feature weren't provided.")
            logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")

            if not _source.has_normals():
                logger.warning(f"Source has no normals which are needed to compute FPFH features.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _source = process_point_cloud(point_cloud=_source,
                                              estimate_normals=True,
                                              search_param_knn=30,
                                              search_param_radius=0.02)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={source: _source})

            _, _source_feature = process_point_cloud(point_cloud=_source,
                                                     compute_feature=True,
                                                     search_param_knn=100,
                                                     search_param_radius=0.05)  # 5cm
        else:
            _source_feature = source_feature

        if target_feature is None:
            logger.warning("Target FPFH feature weren't provided.")
            logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")

            if not _target.has_normals():
                logger.warning(f"Target has no normals which are needed to compute FPFH features.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _target = process_point_cloud(point_cloud=_target,
                                              estimate_normals=True,
                                              search_param_knn=30,
                                              search_param_radius=0.02)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={target: _target})

            _, _target_feature = process_point_cloud(point_cloud=_target,
                                                     compute_feature=True,
                                                     search_param_knn=100,
                                                     search_param_radius=0.05)  # 5cm
        else:
            _target_feature = target_feature

        result = self.algorithm(source=_source,
                                target=_target,
                                source_feature=_source_feature,
                                target_feature=_target_feature,
                                option=self.option)
        if draw:
            self.draw_registration_result(source=_source, target=_target, pose=result.transformation, **kwargs)

        return result


class RANSAC(RegistrationInterface):
    """The *RANSAC* algorithm."""

    def __init__(self,
                 algorithm: Union[ransac_feature, ransac_correspondence] = ransac_feature,
                 max_iteration=100000,
                 confidence=0.999,
                 max_correspondence_distance: float = 0.015,  # 1.5cm
                 estimation_method: EstimationMethodTypes = EstimationMethodTypes.POINT_TO_POINT,
                 checkers: CheckerTypes = (CheckerTypes.EDGE_LENGTH, CheckerTypes.DISTANCE),
                 similarity_threshold: float = 0.9,
                 normal_angle_threshold: float = 0.52,  # ~30° in radians
                 data_to_cache: dict = None,
                 **kwargs: Any) -> None:
        super().__init__(data_to_cache=data_to_cache, **kwargs)

        self.max_iteration = max_iteration
        self.confidence = confidence
        self.max_correspondence_distance = max_correspondence_distance
        self.similarity_threshold = similarity_threshold
        self.normal_angle_threshold = normal_angle_threshold

        self.algorithm = algorithm
        assert estimation_method != EstimationMethodTypes.COLORED, \
            f"Estimation method {estimation_method} is not supported by RANSAC."
        self.estimation_method = eval_estimation_method(estimation_method)

        self.checkers = None
        self._set_checkers(checkers)
        self.criteria = RANSACConvergenceCriteria(max_iteration=self.max_iteration, confidence=self.confidence)

    def _set_checkers(self, checkers: Tuple[CheckerTypes]) -> None:
        self.checkers = list()
        if CheckerTypes.EDGE_LENGTH in checkers:
            self.checkers.append(
                CorrespondenceCheckerBasedOnEdgeLength(similarity_threshold=self.similarity_threshold))
        if CheckerTypes.DISTANCE in checkers:
            self.checkers.append(
                CorrespondenceCheckerBasedOnDistance(distance_threshold=self.max_correspondence_distance))
        if CheckerTypes.NORMAL in checkers:
            self.checkers.append(
                CorrespondenceCheckerBasedOnNormal(normal_angle_threshold=self.normal_angle_threshold))

    def run(self,
            source: InputTypes,
            target: InputTypes,
            source_feature: Feature = None,
            target_feature: Feature = None,
            draw: bool = False,
            **kwargs: Any) -> RegistrationResult:

        _source = self._eval_data(data_key_or_value=source, **kwargs)
        _target = self._eval_data(data_key_or_value=target, **kwargs)

        if any(key in kwargs for key in ["max_iteration", "confidence"]):
            self.criteria = RANSACConvergenceCriteria(
                max_iteration=kwargs.get("max_iteration", self.max_iteration),
                confidence=kwargs.get("confidence", self.confidence))

        self.max_correspondence_distance = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        if self.max_correspondence_distance == -1.0:
            self.max_correspondence_distance = self._compute_dist(point_cloud=_source)

        if "estimation_method" in kwargs:
            estimation_method = kwargs.get("estimation_method", self.estimation_method)
            assert estimation_method != EstimationMethodTypes.COLORED, \
                f"Estimation method {estimation_method} is not supported by RANSAC."
            self.estimation_method = eval_estimation_method(estimation_method)

        if "checkers" in kwargs:
            self._set_checkers(kwargs.get("checkers", self.checkers))

        if isinstance(self.estimation_method,
                      (PointToPlane, ColoredICP)) or source_feature is None or target_feature is None:
            if not _source.has_normals():
                logger.warning(f"Source has no normals which are needed for {self.estimation_method} and FPFH feature.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _source = process_point_cloud(point_cloud=_source,
                                              estimate_normals=True,
                                              search_param_knn=30,
                                              search_param_radius=0.02)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={source: _source})
            if not _target.has_normals():
                logger.warning(f"Target has no normals which are needed for {self.estimation_method} and FPFH feature.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _target = process_point_cloud(point_cloud=_target,
                                              estimate_normals=True,
                                              search_param_knn=30,
                                              search_param_radius=0.02)
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={target: _target})

        if source_feature is None:
            logger.warning("Source FPFH feature weren't provided.")
            logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")
            _, _source_feature = process_point_cloud(point_cloud=_source,
                                                     compute_feature=True,
                                                     search_param_knn=100,
                                                     search_param_radius=0.05)  # 5cm
        else:
            _source_feature = source_feature

        if target_feature is None:
            logger.warning("Target FPFH feature weren't provided.")
            logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")
            _, _target_feature = process_point_cloud(point_cloud=_target,
                                                     compute_feature=True,
                                                     search_param_knn=100,
                                                     search_param_radius=0.05)  # 5cm
        else:
            _target_feature = target_feature

        result = self.algorithm(source=_source,
                                target=_target,
                                source_feature=_source_feature,
                                target_feature=_target_feature,
                                mutual_filter=True,
                                max_correspondence_distance=self.max_correspondence_distance,
                                estimation_method=self.estimation_method,
                                ransac_n=3,
                                checkers=self.checkers,
                                criteria=self.criteria)

        if draw:
            self.draw_registration_result(source=_source, target=_target, pose=result.transformation, **kwargs)

        return result
