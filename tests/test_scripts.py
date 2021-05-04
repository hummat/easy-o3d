"""Integration tests for Easy Open3D."""
import os
import numpy as np
import pytest

from .context import run_registration


@pytest.fixture
def ground_truth_path():
    return "tests/test_data/ground_truth_pose.json"


def test_ground_truth_path(ground_truth_path):
    assert os.path.exists(ground_truth_path)


class TestRunRegistration:

    def test_eval_init_poses_or_ground_truth_string(self, ground_truth_path):
        poses1 = run_registration.eval_init_poses_or_ground_truth(poses=ground_truth_path)
        assert isinstance(poses1, list)
        assert len(poses1) == 1
        assert isinstance(poses1[0], np.ndarray)
        assert poses1[0].shape == (4, 4)

        poses2 = run_registration.eval_init_poses_or_ground_truth(poses=str([ground_truth_path]))
        assert isinstance(poses2, list)
        assert len(poses2) == 1
        assert isinstance(poses2[0], np.ndarray)
        assert poses2[0].shape == (4, 4)

        assert np.all(poses1[0] == poses2[0])

    def test_eval_init_poses_or_ground_truth_float(self):
        poses1 = run_registration.eval_init_poses_or_ground_truth(poses=str([0.0, 0.0, 0.5]))
        assert isinstance(poses1, list)
        assert len(poses1) == 1
        assert isinstance(poses1[0], np.ndarray)
        assert poses1[0].shape == (4, 4)

        poses2 = run_registration.eval_init_poses_or_ground_truth(poses=str([[0, 0, 0.5]]))
        assert isinstance(poses2, list)
        assert len(poses2) == 1
        assert isinstance(poses2[0], np.ndarray)
        assert poses2[0].shape == (4, 4)

        assert np.all(poses1[0] == poses2[0])
