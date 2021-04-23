import copy

import numpy as np
import open3d as o3d

from .context import utils


def test_eval_data_return_types():
    points = np.random.random(size=1000 * 3).reshape(1000, 3)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    image = np.ones(shape=(640, 480)).astype(np.float32)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(image),
                                                                    depth=o3d.geometry.Image(image))
    mesh = o3d.io.read_triangle_mesh(filename="tests/test_data/suzanne.ply")

    assert isinstance(utils.eval_data(data="tests/test_data/suzanne.ply"), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=point_cloud), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=points), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=o3d.geometry.Image(image),
                                      camera_intrinsic=np.eye(3)), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=rgbd_image, camera_intrinsic=np.eye(3)), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data="tests/test_data/suzanne.ply", number_of_points=1000), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=mesh, number_of_points=1000), o3d.geometry.PointCloud)
    assert isinstance(utils.eval_data(data=image, camera_intrinsic=np.eye(3)), o3d.geometry.PointCloud)


class TestProcessPointCloud:

    def test_process_point_cloud_downsample(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

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

    def test_process_point_cloud_remove_outlier(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

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

    def test_process_point_cloud_transform(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

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

    def test_process_point_cloud_scale(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud, scale=0.5)
        assert _point_cloud is not point_cloud
        assert np.asarray(_point_cloud.points).mean() < np.asarray(point_cloud.points).mean()
        assert _point_cloud.get_max_bound().max() <= 0.5

    def test_process_point_cloud_normals(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

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

    def test_process_point_cloud_compute_feature(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

        _point_cloud, feature = utils.process_point_cloud(point_cloud=point_cloud,
                                                          estimate_normals=True,
                                                          compute_feature=True)
        assert _point_cloud is not point_cloud
        assert feature is not None

    def test_process_point_cloud_return_type(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

        point_cloud_orig = copy.deepcopy(point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig)
        assert isinstance(point_cloud, o3d.geometry.PointCloud)
        assert _point_cloud is not point_cloud_orig
