"""Point cloud registration functionality.

Classes:
    CheckerTypes: Supported RANSAC correspondence checker types.
    KernelTypes: Supported ICP Point-To-Plane robust kernel types.
    IterativeClosestPoint: The Iterative Closest Point (ICP) algorithm.
    FastGlobalRegistration: The Fast Global Registration (FGR) algorithm.
    RANSAC: The RANSAC algorithm.
"""
import copy
import logging
import time
from typing import Any, Union, Tuple, List, Dict

import numpy as np
import open3d as o3d

from .interfaces import RegistrationInterface, ICPTypes, MyRegistrationResult
from .utils import InputTypes, process_point_cloud, SearchParamTypes, eval_transformation_data

PointCloud = o3d.geometry.PointCloud
PointToPoint = o3d.pipelines.registration.TransformationEstimationPointToPoint
PointToPlane = o3d.pipelines.registration.TransformationEstimationPointToPlane
ColoredICP = o3d.pipelines.registration.TransformationEstimationForColoredICP
TransformationEstimation = o3d.pipelines.registration.TransformationEstimation
ICPConvergenceCriteria = o3d.pipelines.registration.ICPConvergenceCriteria
RobustKernel = o3d.pipelines.registration.RobustKernel
FastGlobalRegistrationOption = o3d.pipelines.registration.FastGlobalRegistrationOption
Feature = o3d.pipelines.registration.Feature
RANSACConvergenceCriteria = o3d.pipelines.registration.RANSACConvergenceCriteria
CorrespondenceCheckerBasedOnEdgeLength = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength
CorrespondenceCheckerBasedOnDistance = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance
CorrespondenceCheckerBasedOnNormal = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal
CorrespondenceChecker = o3d.pipelines.registration.CorrespondenceChecker
ransac_feature = o3d.pipelines.registration.registration_ransac_based_on_feature_matching
ransac_correspondence = o3d.pipelines.registration.registration_ransac_based_on_correspondence
RegistrationResult = o3d.pipelines.registration.RegistrationResult


class CheckerTypes:
    """Supported RANSAC correspondence checker types."""
    EDGE = CorrespondenceCheckerBasedOnEdgeLength
    DISTANCE = CorrespondenceCheckerBasedOnDistance
    NORMAL = CorrespondenceCheckerBasedOnNormal


class KernelTypes:
    """Supported ICP Point-To-Plane robust kernel types."""
    TUKEY = o3d.pipelines.registration.TukeyLoss
    CAUCHY = o3d.pipelines.registration.CauchyLoss
    L1 = o3d.pipelines.registration.L1Loss
    L2 = o3d.pipelines.registration.L2Loss
    HUBER = o3d.pipelines.registration.HuberLoss
    GM = o3d.pipelines.registration.GMLoss


logger = logging.getLogger(__name__)


class IterativeClosestPoint(RegistrationInterface):
    """The *Iterative Closest Point* (ICP) algorithm.

    The goal is to find the rotation and translation, i.e. 6D pose, of a source object found in the target point cloud
    based on initial pose information.

    Attributes:
        relative_fitness: If relative change (difference) of fitness score is lower than `relative_fitness`, the
                          iteration stops.
        relative_rmse: If relative change (difference) of inliner RMSE is lower than `relative_rmse`, the iteration
                       stops.
        max_iteration: Maximum number of iterations before the algorithm is stopped.
        max_correspondence_distance: Maximum correspondence points-pair distance.
        estimation_method: The estimation method.
        with_scaling: Use non-rigid transformation in Point-to-Point ICP to align source to target.
        kernel: Use robust kernel in Point-to-Plane ICP to deal with noise.
        kernel_noise_std: The estimated/assumed noise standard deviation in the target data used in `kernel`.
        algorithm: The type of ICP registration algorithm used in `run`.

    Methods:
        run(source, target, init, ...): Runs the *Iterative Closest Point* algorithm between `source` and `target` point
                                        cloud with initial pose.
    """
    def __init__(self,
                 relative_fitness: float = 1e-6,
                 relative_rmse: float = 1e-6,
                 max_iteration: int = 30,
                 max_correspondence_distance: float = 0.004,  # 4mm
                 estimation_method: ICPTypes = ICPTypes.POINT,
                 with_scaling: bool = False,
                 kernel: Union[KernelTypes, None] = None,
                 kernel_noise_std: float = 0.1,
                 data_to_cache: Union[Dict[Any, InputTypes], None] = None,
                 auto_cache: bool = True,
                 cache_size: int = 100) -> None:
        """
        Args:
            relative_fitness: If relative change (difference) of fitness score is lower than `relative_fitness`,
                              the iteration stops.
            relative_rmse: If relative change (difference) of inliner RMSE is lower than `relative_rmse`, the iteration
                           stops.
            max_iteration: Maximum number of iterations before the algorithm is stopped.
            max_correspondence_distance: Maximum correspondence points-pair distance.
            estimation_method: The estimation method used by ICP.
            with_scaling: Use non-rigid transformation in Point-to-Point ICP to align source to target.
            kernel: Use robust kernel in Point-to-Plane ICP to deal with noise.
            kernel_noise_std: The estimated/assumed noise standard deviation in the target data used in `kernel`.
            data_to_cache: Data to be cached. Refer to base class for details.
            auto_cache: Automatically cache anything passing trough this function that is not already cached.
            cache_size: Maximum number of elements allowed in cache.
        """
        super().__init__(name="ICP",
                         data_to_cache=data_to_cache,
                         auto_cache=auto_cache,
                         cache_size=cache_size)

        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse
        self.max_iteration = max_iteration
        self.max_correspondence_distance = max_correspondence_distance

        self.algorithm = o3d.pipelines.registration.registration_icp
        self.estimation_method = estimation_method
        if self.estimation_method == ICPTypes.POINT:
            self.name = f"POINT_TO_POINT_{self.name}"
            self._estimation_method = PointToPoint
        elif self.estimation_method == ICPTypes.PLANE:
            self.name = f"POINT_TO_PLANE_{self.name}"
            self._estimation_method = PointToPlane
        elif self.estimation_method == ICPTypes.COLOR:
            self.name = f"COLORED_{self.name}"
            self._estimation_method = ColoredICP
            self.algorithm = o3d.pipelines.registration.registration_colored_icp
        else:
            raise ValueError(f"`estimation_method` must be one of `ICPTypes` but is {self.estimation_method}.")
        self.with_scaling = with_scaling
        self.kernel = kernel
        self.kernel_noise_std = kernel_noise_std

        self.criteria = ICPConvergenceCriteria(relative_fitness=self.relative_fitness,
                                               relative_rmse=self.relative_rmse,
                                               max_iteration=self.max_iteration)

    @staticmethod
    def _crop_target_around_source(source: PointCloud,
                                   target: PointCloud,
                                   init: np.ndarray,
                                   crop_scale: float = 1.0) -> PointCloud:
        """Crops `target` point cloud around `source` point cloud.

        Args:
            source: The source point cloud.
            target: The target point cloud.
            init: The initial pose data for ICP as 4x4 transformation matrix.
            crop_scale: The scale is applied to the source bounding box decrease/increase the cropped area.

        Returns:
            The cropped target point cloud.
        """
        bounding_box = copy.deepcopy(source).transform(init).get_axis_aligned_bounding_box()
        bounding_box = bounding_box.scale(scale=crop_scale, center=bounding_box.get_center())
        # noinspection PyArgumentList
        cropped_target = copy.deepcopy(target).crop(bounding_box=bounding_box)
        return cropped_target if not cropped_target.is_empty() else target

    def _eval_kwargs(self, source: PointCloud, **kwargs: Any) -> None:
        """Evaluates ICP keyword arguments.

        Args:
            source: The source point cloud used to automatically determine `maximum_correspondence_distance`
        """
        if any(key in kwargs for key in ["relative_fitness", "relative_rmse", "max_iteration"]):
            self.criteria = ICPConvergenceCriteria(
                relative_fitness=kwargs.get("relative_fitness", self.relative_fitness),
                relative_rmse=kwargs.get("relative_rmse", self.relative_rmse),
                max_iteration=kwargs.get("max_iteration", self.max_iteration))

        self.max_correspondence_distance = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        if self.max_correspondence_distance == -1.0:
            self.max_correspondence_distance = self._compute_dist(point_cloud=source)

        self.estimation_method = kwargs.get("estimation_method", self.estimation_method)
        if self.estimation_method == ICPTypes.COLOR:
            self._estimation_method = ColoredICP()
            self.algorithm = o3d.pipelines.registration.registration_colored_icp
        elif self.estimation_method == ICPTypes.POINT:
            self._estimation_method = PointToPoint(with_scaling=kwargs.get("with_scaling", self.with_scaling))
        elif self.estimation_method == ICPTypes.PLANE:
            kernel = kwargs.get("kernel", self.kernel)
            if kernel is not None:
                if kernel not in [KernelTypes.L1, KernelTypes.L2]:
                    kernel = kernel(k=kwargs.get("kernel_noise_std", self.kernel_noise_std))
                else:
                    kernel = kernel()
                self._estimation_method = PointToPlane(kernel=kernel)
            else:
                self._estimation_method = PointToPlane()
        else:
            raise ValueError(f"`estimation_method` must be one of `ICPTypes` but is {self.estimation_method}.")

    def run(self,
            source: InputTypes,
            target: InputTypes,
            init: Union[np.ndarray, list, str] = np.eye(4),
            crop_target_around_source: bool = False,
            crop_scale: float = 1.0,
            draw: bool = False,
            **kwargs: Any) -> MyRegistrationResult:
        """Runs the *Iterative Closest Point* (ICP) algorithm between `source` and `target` point cloud.

        The goal is to find the rotation and translation, i.e. 6D pose, of the `source` object, best
        resembling its actual pose found in the `target` point cloud using initial pose information `init`.

        Args:
            source: The source data.
            target: The target data.
            init: The initial pose of `source`. Can be translation, rotation, transformation or "center", in which case
                  `source` is translated to `target` center.
            crop_target_around_source: Crops `target` around the bounding box of `source`. Should only be used if `init`
                                       is already quite accurate.
            crop_scale: The scale of the `source` bounding box used for cropping `target`. Increase if `init` is
                        inaccurate.
            draw: Visualize the registration result.

        Returns:
            The registration result containing relative fitness (`fitness`) and RMSE (`inlier_rmse`) as well as the
            correspondence set between `source` and `target` (`correspondence_set`) and transformation
            (`transformation`) between `source` and `target` and runtime (`runtime`).
        """
        start = time.time()
        _init = eval_transformation_data(init)
        if np.array_equal(_init, np.asarray("center")):
            _init = eval_transformation_data(target.get_center())
        _source = self._eval_data(data_key_or_value=source, **kwargs)
        _target = self._eval_data(data_key_or_value=target, **kwargs)

        if crop_target_around_source:
            _target = self._crop_target_around_source(source=_source, target=_target, init=_init, crop_scale=crop_scale)

        if self.estimation_method in [ICPTypes.PLANE, ICPTypes.COLOR]:
            if not _source.has_normals():
                if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                    logger.warning(f"Source has no normals which are needed to compute FPFH features.")
                    logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _source = process_point_cloud(point_cloud=_source,
                                              estimate_normals=True,
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_knn=kwargs.get("search_param_knn", 30),
                                              search_param_radius=kwargs.get("search_param_radius", 0.02))
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={source: _source})

            if not _target.has_normals():
                if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                    logger.warning(f"Target has no normals which are needed to compute FPFH features.")
                    logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _target = process_point_cloud(point_cloud=_target,
                                              estimate_normals=True,
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_knn=kwargs.get("search_param_knn", 30),
                                              search_param_radius=kwargs.get("search_param_radius", 0.02))
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={target: _target})

        self._eval_kwargs(source=_source, **kwargs)

        # noinspection PyTypeChecker
        result = self.algorithm(source=_source,
                                target=_target,
                                max_correspondence_distance=self.max_correspondence_distance,
                                init=_init,
                                estimation_method=self._estimation_method,
                                criteria=self.criteria)

        runtime = time.time() - start
        logger.debug(f"{self.name} took {runtime} seconds.")
        logger.debug(f"{self.name} result: fitness={result.fitness}, inlier_rmse={result.inlier_rmse}.")

        if draw:
            self.draw_registration_result(source=_source, target=_target, pose=result.transformation, **kwargs)

        return MyRegistrationResult(correspondence_set=result.correspondence_set,
                                    fitness=result.fitness,
                                    inlier_rmse=result.inlier_rmse,
                                    transformation=result.transformation,
                                    runtime=runtime)


class FastGlobalRegistration(RegistrationInterface):
    """The *Fast Global Registration* (FGR) algorithm.

    The goal is to find the rotation and translation, i.e. 6D pose, of a source object found in the target point cloud
    without any initial pose information. As the estimated pose is usually inaccurate, an subsequent ICP refinement
    is commonly employed.

    Attributes:
        max_iteration: Maximum number of iterations before the algorithm is stopped.
        max_correspondence_distance: Maximum correspondence points-pair distance.
        algorithm: The type of FGR registration algorithm used in `run`.
        option: Parameter options object used during execution.

    Methods:
        run(source, target, source_feature, target_feature, ...): Runs the FGR algorithm between `source` and `target`
                                                                  point cloud without any initial pose information.
    """
    def __init__(self,
                 max_iteration: int = 64,
                 max_correspondence_distance: float = 0.005,  # 5mm
                 data_to_cache: Union[Dict[Any, InputTypes], None] = None,
                 auto_cache: bool = True,
                 cache_size: int = 100) -> None:
        """
        Args:
            max_iteration: Maximum number of iterations before the algorithm is stopped.
            max_correspondence_distance: Maximum correspondence points-pair distance.
            data_to_cache: Data to be cached. Refer to base class for details.
            auto_cache: Automatically cache anything passing trough this function that is not already cached.
            cache_size: Maximum number of elements allowed in cache.
        """
        super().__init__(name="FGR",
                         data_to_cache=data_to_cache,
                         auto_cache=auto_cache,
                         cache_size=cache_size)

        self.max_correspondence_distance = max_correspondence_distance
        self.max_iteration = max_iteration
        self.algorithm = o3d.pipelines.registration.registration_fgr_based_on_feature_matching

        self.option = FastGlobalRegistrationOption(maximum_correspondence_distance=self.max_correspondence_distance,
                                                   iteration_number=self.max_iteration)

    def run(self,
            source: InputTypes,
            target: InputTypes,
            init: Union[np.ndarray, list, str] = np.eye(4),
            source_feature: Union[Feature, None] = None,
            target_feature: Union[Feature, None] = None,
            draw: bool = False,
            **kwargs: Any) -> MyRegistrationResult:
        """Runs the Fast Global Registration algorithm between `source` and `target` point cloud.

        The goal is to find the rotation and translation, i.e. 6D pose, of the `source` object, best resembling its
        actual pose found in the `target` point cloud without any initial pose information. The result is commonly
        refined by ICP.

        Args:
            source: The source data.
            target: The target data.
            init: The initial pose of `source`. Can be translation, rotation, transformation or "center", in which case
                  `source` is translated to `target` center.
            source_feature: The FPFH feature of `source`. Computed based on default values if not provided.
            target_feature: The FPFH feature of `target`. Computed based on default values if not provided.
            draw: Visualize the registration result.

        Returns:
            The registration result containing relative fitness (`fitness`) and RMSE (`inlier_rmse`) as well as the
            correspondence set between `source` and `target` (`correspondence_set`) and transformation
            (`transformation`) between `source` and `target` and runtime (`runtime`).
        """
        start = time.time()
        _target = self._eval_data(data_key_or_value=target, **kwargs)
        _init = eval_transformation_data(init)
        if np.array_equal(_init, np.asarray("center")):
            _init = eval_transformation_data(target.get_center())
        _source = copy.deepcopy(self._eval_data(data_key_or_value=source, **kwargs)).transform(_init)

        self.max_correspondence_distance = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        if self.max_correspondence_distance == -1.0:
            self.max_correspondence_distance = self._compute_dist(point_cloud=_source)

        if any(key in kwargs for key in ["max_iteration"]):
            self.option = FastGlobalRegistrationOption(maximum_correspondence_distance=self.max_correspondence_distance,
                                                       iteration_number=kwargs.get("max_iteration", self.max_iteration))

        if source_feature is None:
            if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                logger.warning("Source FPFH feature weren't provided.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")

            if not _source.has_normals():
                if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                    logger.warning(f"Source has no normals which are needed to compute FPFH features.")
                    logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _source = process_point_cloud(point_cloud=_source,
                                              estimate_normals=True,
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_knn=kwargs.get("search_param_knn", 30),
                                              search_param_radius=kwargs.get("search_param_radius", 0.02))
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={source: _source})

            _, _source_feature = process_point_cloud(point_cloud=_source,
                                                     compute_feature=True,
                                                     search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                                     search_param_knn=kwargs.get("search_param_knn", 100),
                                                     search_param_radius=kwargs.get("search_param_radius", 0.05))
        else:
            _source_feature = source_feature

        if target_feature is None:
            if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                logger.warning("Target FPFH feature weren't provided.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")

            if not _target.has_normals():
                if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                    logger.warning(f"Target has no normals which are needed to compute FPFH features.")
                    logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _target = process_point_cloud(point_cloud=_target,
                                              estimate_normals=True,
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_knn=kwargs.get("search_param_knn", 30),
                                              search_param_radius=kwargs.get("search_params_radius", 0.02))
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={target: _target})

            _, _target_feature = process_point_cloud(point_cloud=_target,
                                                     compute_feature=True,
                                                     search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                                     search_param_knn=kwargs.get("search_param_knn", 100),
                                                     search_param_radius=kwargs.get("search_param_radius", 0.05))
        else:
            _target_feature = target_feature

        # noinspection PyTypeChecker
        result = self.algorithm(source=_source,
                                target=_target,
                                source_feature=_source_feature,
                                target_feature=_target_feature,
                                option=self.option)

        runtime = time.time() - start
        logger.debug(f"{self.name} took {runtime} seconds.")
        logger.debug(f"{self.name} result: fitness={result.fitness}, inlier_rmse={result.inlier_rmse}.")

        if draw:
            self.draw_registration_result(source=_source, target=_target, pose=result.transformation, **kwargs)

        return MyRegistrationResult(correspondence_set=result.correspondence_set,
                                    fitness=result.fitness,
                                    inlier_rmse=result.inlier_rmse,
                                    transformation=result.transformation @ _init,
                                    runtime=runtime)


class RANSAC(RegistrationInterface):
    """The *RANSAC* algorithm.

    The goal is to find the rotation and translation, i.e. 6D pose, of a source object found in the target point cloud
    without any initial pose information. As the estimated pose is usually inaccurate, an subsequent ICP refinement
    is commonly employed. Older, but well established alternative to FGR.

    Attributes:
        algorithm: The type of RANSAC registration algorithm used in `run`.
        max_iteration: Maximum number of iterations before the algorithm is stopped.
        confidence: Threshold for algorithm convergence based on point-pair correspondence confidence.
        max_correspondence_distance: Maximum correspondence points-pair distance.
        estimation_method: The estimation method used by RANSAC.
        with_scaling: Use non-rigid transformation in Point-to-Point estimation method to align source to target.
        kernel: Use robust kernel in Point-to-Plane estimation method to deal with noise.
        kernel_noise_std: The estimated/assumed noise standard deviation in the target data used in `kernel`.
        ransac_n: Number of point-pair correspondences used for alignment.
        checkers: The correspondence checkers used to discard correspondences using low compute metrics.
        similarity_threshold: Edge length similarity checker threshold.
        normal_angle_threshold: Normal angle similarity checker threshold.
        criteria: RANSAC criteria object specifying algorithm convergence thresholds.

    Methods:
        run(source, target, source_feature, target_feature, ...): Runs the RANSAC algorithm between `source` and
                                                                  `target` point cloud and their corresponding FPFH
                                                                  features without any initial pose information.
        _eval_checkers(): Constructs point-pair correspondence checker list from given checker types list.
    """
    def __init__(self,
                 algorithm: Union[ransac_feature, ransac_correspondence] = ransac_feature,
                 max_iteration: int = 100000,
                 confidence: float = 0.999,
                 max_correspondence_distance: float = 0.015,  # 1.5cm
                 estimation_method: ICPTypes = ICPTypes.POINT,
                 with_scaling: bool = False,
                 kernel: Union[KernelTypes, None] = None,
                 kernel_noise_std: float = 0.1,
                 ransac_n: int = 3,
                 checkers: Union[List[CheckerTypes], Tuple[CheckerTypes]] = (CheckerTypes.EDGE, CheckerTypes.DISTANCE),
                 similarity_threshold: float = 0.9,
                 normal_angle_threshold: float = 30.0,
                 data_to_cache: Union[Dict[Any, InputTypes], None] = None,
                 auto_cache: bool = True,
                 cache_size: int = 100) -> None:
        """
        Args:
            algorithm: The type of RANSAC registration algorithm used in `run`.
            max_iteration: Maximum number of iterations before the algorithm is stopped.
            confidence: Threshold for algorithm convergence based on point-pair correspondence confidence.
            max_correspondence_distance: Maximum correspondence points-pair distance.
            estimation_method: The estimation method used by RANSAC.
            with_scaling: Use non-rigid transformation in Point-to-Point estimation method to align source to target.
            kernel: Use robust kernel in Point-to-Plane estimation method to deal with noise.
            kernel_noise_std: The estimated/assumed noise standard deviation in the target data used in `kernel`.
            ransac_n: Number of point-pair correspondences used for alignment.
            checkers: The correspondence checkers used to discard correspondences using low compute metrics.
            similarity_threshold: Edge length similarity checker threshold.
            normal_angle_threshold: Normal angle similarity checker threshold in degrees.
            data_to_cache: Data to be cached. Refer to base class for details.
            auto_cache: Automatically cache anything passing trough this function that is not already cached.
            cache_size: Maximum number of elements allowed in cache.
        """
        super().__init__(name="RANSAC",
                         data_to_cache=data_to_cache,
                         auto_cache=auto_cache,
                         cache_size=cache_size)

        self.max_iteration = max_iteration
        self.confidence = confidence
        self.max_correspondence_distance = max_correspondence_distance
        self.similarity_threshold = similarity_threshold
        self.normal_angle_threshold = np.deg2rad(normal_angle_threshold)
        self.ransac_n = ransac_n

        self.algorithm = algorithm
        self.estimation_method = estimation_method
        if self.estimation_method == ICPTypes.COLOR:
            raise ValueError(f"Estimation method {self.estimation_method} is not supported by RANSAC.")
        elif self.estimation_method == ICPTypes.POINT:
            self._estimation_method = PointToPoint
        elif self.estimation_method == ICPTypes.PLANE:
            self._estimation_method = PointToPlane
        else:
            raise ValueError(f"`estimation_method` must be one of `ICPTypes` but is {type(estimation_method)}.")
        self.with_scaling = with_scaling
        self.kernel = kernel
        self.kernel_noise_std = kernel_noise_std

        self.checkers = checkers
        self._checkers = self._eval_checkers(checkers=self.checkers)
        self.criteria = RANSACConvergenceCriteria(max_iteration=self.max_iteration, confidence=self.confidence)

    def _eval_checkers(self, **kwargs: Any) -> Union[List[CorrespondenceChecker], List]:
        """Constructs point-pair correspondence checker list from given checker types list.

        Returns:
            List of point-pair correspondence checkers used during RANSAC execution.
        """
        checkers = kwargs.get("checkers")
        if isinstance(checkers, (list, tuple)):
            checker_list = list()
            for checker in checkers:
                if checker == CheckerTypes.EDGE:
                    checker_list.append(
                        checker(similarity_threshold=kwargs.get("similarity_threshold", self.similarity_threshold)))
                elif checker == CheckerTypes.DISTANCE:
                    checker_list.append(checker(
                        distance_threshold=kwargs.get("max_correspondence_distance", self.max_correspondence_distance)))
                elif checker == CheckerTypes.NORMAL:
                    checker_list.append(checker(
                        normal_angle_threshold=kwargs.get("normal_angle_threshold", self.normal_angle_threshold)))
            return checker_list
        return self._checkers if self._checkers else list()

    def run(self,
            source: InputTypes,
            target: InputTypes,
            init: Union[np.ndarray, list, str] = np.eye(4),
            source_feature: Union[Feature, None] = None,
            target_feature: Union[Feature, None] = None,
            draw: bool = False,
            **kwargs: Any) -> MyRegistrationResult:
        """Runs the RANSAC algorithm between `source` and `target` point cloud.

        The goal is to find the rotation and translation, i.e. 6D pose, of the `source` object, best resembling its
        actual pose found in the `target` point cloud without any initial pose information. The result is commonly
        refined by ICP.

        Args:
            source: The source data.
            target: The target data.
            init: The initial pose of `source`. Can be translation, rotation, transformation or "center", in which case
                  `source` is translated to `target` center.
            source_feature: The FPFH feature of `source`. Computed based on default values if not provided.
            target_feature: The FPFH feature of `target`. Computed based on default values if not provided.
            draw: Visualize the registration result.

        Returns:
            The registration result containing relative fitness (`fitness`) and RMSE (`inlier_rmse`) as well as the
            correspondence set between `source` and `target` (`correspondence_set`) and transformation
            (`transformation`) between `source` and `target` and runtime (`runtime`).
        """
        start = time.time()
        _target = self._eval_data(data_key_or_value=target, **kwargs)
        _init = eval_transformation_data(init)
        if np.array_equal(_init, np.asarray("center")):
            _init = eval_transformation_data(target.get_center())
        _source = copy.deepcopy(self._eval_data(data_key_or_value=source, **kwargs)).transform(_init)

        if any(key in kwargs for key in ["max_iteration", "confidence"]):
            self.criteria = RANSACConvergenceCriteria(
                max_iteration=kwargs.get("max_iteration", self.max_iteration),
                confidence=kwargs.get("confidence", self.confidence))

        self.max_correspondence_distance = kwargs.get("max_correspondence_distance", self.max_correspondence_distance)
        if self.max_correspondence_distance == -1.0:
            self.max_correspondence_distance = self._compute_dist(point_cloud=_source)
            self._checkers = self._eval_checkers(checkers=self.checkers)

        self.estimation_method = kwargs.get("estimation_method", self.estimation_method)
        if self.estimation_method == ICPTypes.COLOR:
            raise ValueError(f"Estimation method {self.estimation_method} is not supported by RANSAC.")
        elif self.estimation_method == ICPTypes.POINT:
            self._estimation_method = PointToPoint(with_scaling=kwargs.get("with_scaling", self.with_scaling))
        elif self.estimation_method == ICPTypes.PLANE:
            kernel = kwargs.get("kernel", self.kernel)
            if kernel is not None:
                if kernel not in [KernelTypes.L1, KernelTypes.L2]:
                    kernel = kernel(k=kwargs.get("kernel_noise_std", self.kernel_noise_std))
                else:
                    kernel = kernel()
                self._estimation_method = PointToPlane(kernel=kernel)
            else:
                self._estimation_method = PointToPlane()
        else:
            raise ValueError(f"`estimation_method` must be one of `ICPTypes` but is {self.estimation_method}.")

        if source_feature is None:
            if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                logger.warning("Source FPFH feature weren't provided.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")

            if not _source.has_normals():
                if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                    logger.warning(f"Source has no normals which are needed to compute FPFH features.")
                    logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _source = process_point_cloud(point_cloud=_source,
                                              estimate_normals=True,
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_knn=kwargs.get("search_param_knn", 30),
                                              search_param_radius=kwargs.get("search_param_radius", 0.02))
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={source: _source})

            _, _source_feature = process_point_cloud(point_cloud=_source,
                                                     compute_feature=True,
                                                     search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                                     search_param_knn=kwargs.get("search_param_knn", 100),
                                                     search_param_radius=kwargs.get("search_param_radius", 0.05))
        else:
            _source_feature = source_feature

        if target_feature is None:
            if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                logger.warning("Target FPFH feature weren't provided.")
                logger.warning("Computing with (potentially suboptimal) default parameters: kNN=100, radius=0.05.")

            if not _target.has_normals():
                if "search_param_knn" not in kwargs and "search_param_radius" not in kwargs:
                    logger.warning(f"Target has no normals which are needed to compute FPFH features.")
                    logger.warning("Computing with (potentially suboptimal) default parameters: kNN=30, radius=0.02.")
                _target = process_point_cloud(point_cloud=_target,
                                              estimate_normals=True,
                                              search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                              search_param_knn=kwargs.get("search_param_knn", 30),
                                              search_param_radius=kwargs.get("search_params_radius", 0.02))
                # If cached before, replace with new version with estimated normals
                self.replace_in_cache(data={target: _target})

            _, _target_feature = process_point_cloud(point_cloud=_target,
                                                     compute_feature=True,
                                                     search_param=kwargs.get("search_param", SearchParamTypes.HYBRID),
                                                     search_param_knn=kwargs.get("search_param_knn", 100),
                                                     search_param_radius=kwargs.get("search_param_radius", 0.05))
        else:
            _target_feature = target_feature

        # noinspection PyTypeChecker
        result = self.algorithm(source=_source,
                                target=_target,
                                source_feature=_source_feature,
                                target_feature=_target_feature,
                                mutual_filter=kwargs.get("mutual_filter", True),
                                max_correspondence_distance=self.max_correspondence_distance,
                                estimation_method=self._estimation_method,
                                ransac_n=kwargs.get("ransac_n", self.ransac_n),
                                checkers=self._eval_checkers(**kwargs),
                                criteria=self.criteria)

        runtime = time.time() - start
        logger.debug(f"{self.name} took {runtime} seconds.")
        logger.debug(f"{self.name} result: fitness={result.fitness}, inlier_rmse={result.inlier_rmse}.")

        if draw:
            self.draw_registration_result(source=_source,
                                          target=_target,
                                          pose=result.transformation,
                                          **kwargs)

        return MyRegistrationResult(correspondence_set=result.correspondence_set,
                                    fitness=result.fitness,
                                    inlier_rmse=result.inlier_rmse,
                                    transformation=result.transformation @ _init,
                                    runtime=runtime)
