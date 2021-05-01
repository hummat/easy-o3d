"""Unittests for the utils module."""

import copy

import numpy as np
import open3d as o3d
import pytest

from .context import utils


@pytest.fixture
def points():
    return np.random.random(size=1000 * 3).reshape(1000, 3)


@pytest.fixture
def point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


@pytest.fixture
def image():
    return np.ones(shape=(640, 480)).astype(np.float32)


# noinspection PyTypeChecker
@pytest.fixture
def rgbd_image(image):
    return o3d.geometry.RGBDImage().create_from_color_and_depth(color=o3d.geometry.Image(image),
                                                                depth=o3d.geometry.Image(image))


@pytest.fixture
def source_path():
    return "tests/test_data/suzanne.ply"


@pytest.fixture
def ground_truth_path():
    return "tests/test_data/ground_truth_pose.json"


@pytest.fixture
def blenderproc_bopwriter_ground_truth_path():
    return "tests/test_data/bop_data/obj_of_interest/train_pbr/000000/scene_gt.json"


@pytest.fixture
def transformation_rotation_matrix():
    return [[0.5099551677703857, 0.25742393732070923, -0.8207793831825256,
             -0.8590373396873474, 0.10278679430484772, -0.5014877915382385,
             -0.044729556888341904, 0.9608163237571716, 0.27355334162712097],
            [239.34751892089844, 255.32371520996094, 1949.2823486328125]]


@pytest.fixture
def transformation_euler():
    return [[10.0, 30.0, 50.0], [100.0, 20.0, 50.0]]


@pytest.fixture
def transformation_quaternion():
    return [[0.925, 0.325, 0.125, -0.035], [0.083, 0.123, 0.567]]


@pytest.fixture
def mesh(source_path):
    return o3d.io.read_triangle_mesh(filename=source_path)


def test_eval_data_return_types(point_cloud, points, image, rgbd_image, mesh, source_path):
    assert isinstance(utils.eval_data(data=source_path), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=point_cloud), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=points), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=o3d.geometry.Image(image),
                                      camera_intrinsic=np.eye(3)), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=rgbd_image, camera_intrinsic=np.eye(3)), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=source_path, number_of_points=1000), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=mesh, number_of_points=1000), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=image, camera_intrinsic=np.eye(3)), o3d.geometry.PointCloud)


class TestProcessPointCloud:

    def test_process_point_cloud_downsample(self, point_cloud):
        # Voxel
        point_cloud_orig = copy.deepcopy(point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig,
                                                 downsample=utils.DownsampleTypes.VOXEL,
                                                 downsample_factor=2.0)
        assert _point_cloud is not point_cloud_orig
        assert np.asarray(point_cloud_orig.points).size != np.asarray(_point_cloud.points).size

        # Uniform
        point_cloud_orig = copy.deepcopy(point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig,
                                                 downsample=utils.DownsampleTypes.UNIFORM,
                                                 downsample_factor=2)
        assert _point_cloud is not point_cloud_orig
        assert np.asarray(point_cloud_orig.points).size // 2 == np.asarray(_point_cloud.points).size

    def test_process_point_cloud_remove_outlier(self, point_cloud):
        # Statistical
        point_cloud_orig = copy.deepcopy(point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig,
                                                 remove_outlier=utils.OutlierTypes.STATISTICAL,
                                                 outlier_std_ratio=0.1,
                                                 search_param_knn=5)
        assert _point_cloud is not point_cloud_orig
        assert len(np.asarray(_point_cloud.points))
        assert len(np.asarray(point_cloud_orig.points)) != len(np.asarray(_point_cloud.points))

        # Radius
        point_cloud_orig = copy.deepcopy(point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig,
                                                 remove_outlier=utils.OutlierTypes.RADIUS,
                                                 search_param_knn=3,
                                                 search_param_radius=0.1)
        assert _point_cloud is not point_cloud_orig
        assert len(np.asarray(_point_cloud.points))
        assert len(np.asarray(point_cloud_orig.points)) != len(np.asarray(_point_cloud.points))

    def test_process_point_cloud_transform(self, point_cloud):
        # Translation
        point_cloud_orig = copy.deepcopy(point_cloud)
        t1 = np.array([1.3, 2.0, 5.7])
        t2 = np.array([1.3, 2.0, 5.7, 1.0])
        t3 = np.array([[1.3, 2.0, 5.7, 1.0]])
        for t in [t1, t2, t3]:
            _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig, transform=t)
            assert np.allclose(_point_cloud.get_center(), point_cloud_orig.get_center() + t.ravel()[:3])
            assert _point_cloud is not point_cloud_orig

        # Rotation
        point_cloud_orig = copy.deepcopy(point_cloud)
        r1 = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        r2 = np.array(r1)
        r3 = r2.flatten().tolist()
        for r in [r1, r2, r3]:
            _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig, transform=r)
            assert _point_cloud is not point_cloud_orig
            assert not np.allclose(np.asarray(_point_cloud.points), np.asarray(point_cloud_orig.points))

    def test_process_point_cloud_scale(self, point_cloud):
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud, scale=0.5)
        assert _point_cloud is not point_cloud
        assert np.asarray(_point_cloud.points).mean() < np.asarray(point_cloud.points).mean()
        assert _point_cloud.get_max_bound().max() <= 0.5

    def test_process_point_cloud_normals(self, point_cloud):
        # Estimate
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud, estimate_normals=True)
        assert _point_cloud is not point_cloud
        assert _point_cloud.has_normals()

        # Normalize
        point_cloud = copy.deepcopy(_point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud, normalize_normals=True)
        assert _point_cloud is not point_cloud
        assert all([np.isclose(np.linalg.norm(n), 1.0) for n in np.asarray(_point_cloud.normals)])

        # Orient
        point_cloud = copy.deepcopy(_point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud,
                                                 orient_normals=utils.OrientationTypes.CAMERA)
        assert _point_cloud is not point_cloud
        assert not np.allclose(np.asarray(_point_cloud.normals), np.asarray(point_cloud.normals))

    def test_process_point_cloud_compute_feature(self, point_cloud):
        _point_cloud, feature = utils.process_point_cloud(point_cloud=point_cloud,
                                                          estimate_normals=True,
                                                          compute_feature=True)
        assert _point_cloud is not point_cloud
        assert feature is not None

    def test_process_point_cloud_return_type(self, point_cloud):
        point_cloud_orig = copy.deepcopy(point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig)
        assert isinstance(point_cloud, o3d.geometry.PointCloud)
        assert _point_cloud is not point_cloud_orig


class TestEvalTransformationData:

    def test_eval_transformation_data_with_transformation_list(self):
        T = np.eye(4)
        R = T[:4, :3].ravel().tolist()
        t = T[:, 3].ravel().tolist()
        T = utils.eval_transformation_data(transformation_data=R + t)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert np.all(T == np.eye(4))

    def test_eval_transformation_data_with_identity_transformation(self):
        T = utils.eval_transformation_data(transformation_data=np.eye(4))
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert np.all(T == np.eye(4))

    def test_eval_transformation_data_with_rotation_matrix(self, transformation_rotation_matrix):
        T = utils.eval_transformation_data(transformation_data=transformation_rotation_matrix)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :] == [0, 0, 0, 1])
        assert all(T[:3, :3].ravel() == transformation_rotation_matrix[0])
        assert all(T[:3, 3].ravel() == transformation_rotation_matrix[1])

        R = transformation_rotation_matrix[0]
        t = transformation_rotation_matrix[1]
        T = utils.eval_transformation_data(transformation_data=R + t)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :] == [0, 0, 0, 1])
        assert all(T[:3, :3].ravel() == transformation_rotation_matrix[0])
        assert all(T[:3, 3].ravel() == transformation_rotation_matrix[1])

        R = transformation_rotation_matrix[0] + [0, 0, 0]
        t = transformation_rotation_matrix[1] + [1]
        T = np.hstack([np.asarray(R).reshape(4, 3), np.asarray(t).reshape(4, 1)])
        T = utils.eval_transformation_data(transformation_data=T)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :] == [0, 0, 0, 1])
        assert all(T[:3, :3].ravel() == transformation_rotation_matrix[0])
        assert all(T[:3, 3].ravel() == transformation_rotation_matrix[1])

    def test_eval_transformation_data_with_euler_angles(self, transformation_euler):
        T = utils.eval_transformation_data(transformation_data=transformation_euler)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :].ravel() == [0, 0, 0, 1])
        assert all(T[:3, 3].ravel() == transformation_euler[1])

        R = transformation_euler[0]
        t = transformation_euler[1]
        T = utils.eval_transformation_data(transformation_data=R + t)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :].ravel() == [0, 0, 0, 1])
        assert all(T[:3, 3].ravel() == t)

    def test_eval_transformation_data_with_quaternion(self, transformation_quaternion):
        T = utils.eval_transformation_data(transformation_data=transformation_quaternion)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :].ravel() == [0, 0, 0, 1])
        assert all(T[:3, 3].ravel() == transformation_quaternion[1])

        R = transformation_quaternion[0]
        t = transformation_quaternion[1]
        T = utils.eval_transformation_data(transformation_data=R + t)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :].ravel() == [0, 0, 0, 1])
        assert all(T[:3, 3].ravel() == t)

    def test_eval_transform_data_with_json(self, ground_truth_path):
        T = utils.eval_transformation_data(transformation_data=ground_truth_path)
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert all(T[3, :].ravel() == [0, 0, 0, 1])
