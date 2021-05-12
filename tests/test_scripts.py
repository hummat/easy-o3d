"""Integration tests for the package scripts."""
import os
import pytest
import configparser

from .context import run_registration


@pytest.fixture
def registration_ini_path():
    return "scripts/registration.ini"


@pytest.fixture
def ground_truth_path():
    return "tests/test_data/ground_truth_pose.json"


def test_paths(registration_ini_path, ground_truth_path):
    assert os.path.exists(registration_ini_path)
    assert os.path.exists(ground_truth_path)


class TestRunRegistration:

    def test_eval_config(self, registration_ini_path):
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(registration_ini_path)
        run_registration.eval_config(config)

    def test_run(self, registration_ini_path):
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(registration_ini_path)
        run_registration.run(config)
