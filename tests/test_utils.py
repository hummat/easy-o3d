import copy

import numpy as np
import open3d as o3d

from .context import utils


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

    def test_process_point_cloud_normals(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud, estimate_normals=True)
        assert _point_cloud is not point_cloud
        assert _point_cloud.has_normals()

        point_cloud = copy.deepcopy(_point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud, normalize_normals=True)
        assert _point_cloud is not point_cloud
        assert all([np.isclose(np.linalg.norm(n), 1.0) for n in np.asarray(_point_cloud.normals)])

    def test_process_point_cloud_return_type(self):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.random.random(size=1000 * 3).reshape(1000, 3))

        point_cloud_orig = copy.deepcopy(point_cloud)
        _point_cloud = utils.process_point_cloud(point_cloud=point_cloud_orig)
        assert isinstance(point_cloud, o3d.geometry.PointCloud)
        assert _point_cloud is not point_cloud_orig
