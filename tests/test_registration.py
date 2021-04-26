"""Integration tests for Easy Open3D."""

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
def icp_data(global_data):
    source, target = global_data
    source.translate([0, 0, 0.5])
    return source, target


class TestIterativeClosestPoint:

    def test_iterative_closest_point_constructor(self):
        registration.IterativeClosestPoint()

    def test_iterative_closest_point_basic_run(self, icp_data):
        icp = registration.IterativeClosestPoint()
        result = icp.run(*icp_data)
        assert not np.allclose(result.transformation, np.eye(4))

    def test_iterative_closest_point_point_to_plane_run(self, icp_data):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.PLANE)
        result = icp.run(*icp_data)
        assert not np.allclose(result.transformation, np.eye(4))

    def test_iterative_closest_point_colored_icp_run(self, icp_data):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.COLOR)
        result = icp.run(*icp_data)
        assert not np.allclose(result.transformation, np.eye(4))


class TestRANSAC:

    def test_iterative_closest_point_constructor(self):
        registration.RANSAC()

    def test_ransac_basic_run(self, global_data):
        ransac = registration.RANSAC()
        result = ransac.run(*global_data)
        assert not np.allclose(result.transformation, np.eye(4))


class TestFastGlobalRegistration:

    def test_iterative_closest_point_constructor(self):
        registration.FastGlobalRegistration()

    def test_ransac_basic_run(self, global_data):
        fgr = registration.FastGlobalRegistration()
        result = fgr.run(*global_data)
        assert not np.allclose(result.transformation, np.eye(4))
