"""Integration tests for Easy Open3D."""

import json
import numpy as np
import pytest

from .context import registration, utils


@pytest.fixture
def source_path():
    return "tests/test_data/suzanne.ply"


@pytest.fixture
def target_path():
    return "tests/test_data/suzanne_on_chair.ply"


@pytest.fixture
def global_data(source_path, target_path):
    source = utils.eval_data(data=source_path, number_of_points=10000)
    target = utils.eval_data(data=target_path, number_of_points=100000)
    return source, target


@pytest.fixture
def feature(global_data):
    source, target = global_data
    _, source_feature = utils.process_point_cloud(point_cloud=source,
                                                  compute_feature=True,
                                                  search_param_knn=100,
                                                  search_param_radius=0.05)
    _, target_feature = utils.process_point_cloud(point_cloud=target,
                                                  compute_feature=True,
                                                  search_param_knn=100,
                                                  search_param_radius=0.05)
    return source_feature, target_feature


@pytest.fixture
def ground_truth():
    with open("tests/test_data/ground_truth_pose.json") as f:
        gt_pose_data = json.load(f)
    rotation = gt_pose_data["rotation_quaternion"]
    translation = gt_pose_data["translation_xyz"]
    return utils.get_transformation_matrix_from_quaternion(rotation_wxyz=rotation, translation_xyz=translation)


class TestIterativeClosestPoint:

    def test_iterative_closest_point_constructor(self):
        registration.IterativeClosestPoint()

    def test_iterative_closest_point_basic(self, global_data):
        icp = registration.IterativeClosestPoint()
        icp.run(*global_data)

    def test_iterative_closest_point_point_to_point(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint(max_iteration=300, max_correspondence_distance=0.1)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.1

    def test_iterative_closest_point_point_to_point_with_scaling(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint(with_scaling=True)
        result = icp.run(*global_data, init=ground_truth)
        assert np.linalg.norm(result.transformation - ground_truth) < 0.01

    def test_iterative_closest_point_point_to_plane(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.PLANE,
                                                 max_iteration=100,
                                                 max_correspondence_distance=0.05)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.1

    def test_iterative_closest_point_point_to_plane_with_robust_kernel(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.PLANE,
                                                 max_iteration=100,
                                                 max_correspondence_distance=0.05,
                                                 kernel=registration.KernelTypes.TUKEY)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.1

    def test_iterative_closest_point_color(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.COLOR,
                                                 max_correspondence_distance=0.4)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.1

    def test_iterative_closest_point_icp_point_to_point_multi_scale(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint()
        result = icp.run_multi_scale(*global_data,
                                     init=[0, 0, 0.5],
                                     source_scales=[0.02, 0.01, 0.005],
                                     iterations=[300, 200, 100],
                                     radius_multiplier=[3, 5])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.05

    def test_iterative_closest_point_icp_point_to_plane_multi_scale(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.PLANE)
        result = icp.run_multi_scale(*global_data,
                                     init=[0.1, 0.1, 0.5],
                                     iterations=[200, 100, 50])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.05

    def test_iterative_closest_point_icp_color_multi_scale(self, global_data, ground_truth):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.COLOR)
        result = icp.run_multi_scale(*global_data,
                                     init=[0, 0, 0.5],
                                     source_scales=[0.02, 0.01, 0.005])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.05


class TestRANSAC:

    def test_ransac_constructor(self):
        registration.RANSAC()

    def test_ransac_basic(self, global_data):
        ransac = registration.RANSAC()
        ransac.run(*global_data)

    def test_ransac_point_to_point(self, global_data, feature, ground_truth):
        source, target = global_data
        source_feature, target_feature = feature
        ransac = registration.RANSAC()
        result = ransac.run(source=source,
                            target=target,
                            source_feature=source_feature,
                            target_feature=target_feature)
        assert np.linalg.norm(result.transformation - ground_truth) < 0.5

    def test_ransac_point_to_plane(self, global_data, feature, ground_truth):
        source, target = global_data
        source_feature, target_feature = feature
        ransac = registration.RANSAC(estimation_method=registration.ICPTypes.PLANE,
                                     max_correspondence_distance=1.0)
        result = ransac.run(source=source,
                            target=target,
                            source_feature=source_feature,
                            target_feature=target_feature)
        assert not np.allclose(result.transformation, np.eye(4))

    def test_ransac_multi_scale(self, global_data, ground_truth):
        source, target = global_data
        ransac = registration.RANSAC()
        result = ransac.run_multi_scale(source=source,
                                        target=target,
                                        source_scales=[0.02, 0.01, 0.005],
                                        iterations=[100000, 50000, 20000],
                                        radius_multiplier=[2, 3])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.5


class TestFastGlobalRegistration:

    def test_fgr_constructor(self):
        registration.FastGlobalRegistration()

    def test_fgr_basic(self, global_data):
        fgr = registration.FastGlobalRegistration()
        fgr.run(*global_data)

    def test_fgr(self, global_data, feature, ground_truth):
        source, target = global_data
        source_feature, target_feature = feature
        fgr = registration.FastGlobalRegistration()
        result = fgr.run(source=source,
                         target=target,
                         source_feature=source_feature,
                         target_feature=target_feature)
        assert np.linalg.norm(result.transformation - ground_truth) < 0.5

    def test_fgr_multi_scale(self, global_data, ground_truth):
        source, target = global_data
        fgr = registration.FastGlobalRegistration()
        result = fgr.run_multi_scale(source=source,
                                     target=target,
                                     source_scales=[0.02, 0.01, 0.005])
        assert np.linalg.norm(result.transformation - ground_truth) < 0.5
