"""Integration tests for Easy Open3D."""
import numpy as np
import pytest
import copy

from .context import registration, utils


@pytest.fixture
def scene_id():
    return 20


@pytest.fixture
def source_path():
    return "tests/test_data/suzanne.ply"


@pytest.fixture
def target_path():
    return "tests/test_data/suzanne_on_chair.ply"


@pytest.fixture
def color_path(scene_id):
    return f"tests/test_data/bop_data/obj_of_interest/train_pbr/000000/rgb/{str(scene_id).zfill(6)}.png"


@pytest.fixture
def depth_path(scene_id):
    return f"tests/test_data/bop_data/obj_of_interest/train_pbr/000000/depth/{str(scene_id).zfill(6)}.png"


@pytest.fixture
def scene_camera_path():
    return "tests/test_data/bop_data/obj_of_interest/train_pbr/000000/scene_camera.json"


@pytest.fixture
def camera_path():
    return "tests/test_data/bop_data/obj_of_interest/camera.json"


@pytest.fixture
def camera_parameters(scene_id, scene_camera_path, camera_path):
    return utils.get_camera_parameters_from_blenderproc_bopwriter(scene_id=scene_id,
                                                                  path_to_scene_camera_json=scene_camera_path,
                                                                  path_to_camera_json=camera_path)


@pytest.fixture
def camera_intrinsic(camera_parameters):
    return camera_parameters.intrinsic


@pytest.fixture
def camera_extrinsic(camera_parameters):
    return camera_parameters.extrinsic


@pytest.fixture
def source_from_mesh(source_path):
    return utils.eval_data(data=source_path, number_of_points=10000)


@pytest.fixture
def target_from_mesh(target_path):
    return utils.eval_data(data=target_path, number_of_points=100000)


@pytest.fixture
def target_from_rgbd(color_path, depth_path, camera_parameters):
    return utils.eval_data(data=[color_path, depth_path],
                           camera_intrinsic=camera_parameters.intrinsic,
                           camera_extrinsic=camera_parameters.extrinsic,
                           depth_trunc=2.0)


@pytest.fixture
def global_data(source_from_mesh, target_from_mesh):
    return source_from_mesh, target_from_mesh


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
def ground_truth_path():
    return "tests/test_data/ground_truth_pose.json"


@pytest.fixture
def ground_truth_world(ground_truth_path):
    return utils.eval_transformation_data(transformation_data=ground_truth_path)


@pytest.fixture
def scene_gt_path():
    return "tests/test_data/bop_data/obj_of_interest/train_pbr/000000/scene_gt.json"


@pytest.fixture
def ground_truth_camera(scene_gt_path, scene_id):
    return utils.get_ground_truth_pose_from_blenderproc_bopwriter(path_to_scene_gt_json=scene_gt_path,
                                                                  scene_id=scene_id)


class TestIterativeClosestPoint:

    def test_iterative_closest_point_constructor(self):
        registration.IterativeClosestPoint()

    def test_iterative_closest_point_eval_data(self, source_path, source_from_mesh):
        icp = registration.IterativeClosestPoint(data_to_cache={"suzanne_str": source_path,
                                                                "suzanne_pcd": source_from_mesh},
                                                 cache_size=10)
        assert icp.is_in_cache(data_key_or_value="suzanne_str")
        assert icp.is_in_cache(data_key_or_value="suzanne_pcd")
        assert isinstance(icp.get_cache_value(data_key="suzanne_str"), registration.PointCloud)
        assert isinstance(icp.get_cache_value(data_key="suzanne_pcd"), registration.PointCloud)

        assert icp._eval_data(data_key_or_value="suzanne_str") is icp.get_cache_value(data_key="suzanne_str")
        assert icp._eval_data(data_key_or_value="suzanne_pcd") is icp.get_cache_value(data_key="suzanne_pcd")
        assert icp.get_cache_value(data_key="suzanne_pcd") is source_from_mesh
        assert len(icp._cached_data) == 2

        icp._eval_data(data_key_or_value=source_from_mesh)
        assert icp.is_in_cache(data_key_or_value=source_from_mesh)
        assert isinstance(icp.get_cache_value(data_key=icp.get_cache_key(source_from_mesh)), registration.PointCloud)
        assert len(icp._cached_data) == 2

        icp._eval_data(data_key_or_value=source_path)
        assert icp.is_in_cache(source_path)
        assert isinstance(icp.get_cache_value(data_key=source_path), registration.PointCloud)
        assert len(icp._cached_data) == 3

        for _ in range(10):
            icp._eval_data(data_key_or_value=source_path)
            icp._eval_data(data_key_or_value=source_from_mesh)
        assert len(icp._cached_data) == 3

        icp._eval_data(data_key_or_value=copy.deepcopy(source_from_mesh))
        assert len(icp._cached_data) == 3

        for i in range(10):
            icp.add_to_cache(data={i: source_from_mesh})
        assert len(icp._cached_data) == 3

        for i in range(10):
            icp.add_to_cache(data={i: source_from_mesh}, replace=False)
        assert len(icp._cached_data) == 3

        for i in range(8):
            icp.add_to_cache(data={i: copy.deepcopy(source_from_mesh)})
        assert len(icp._cached_data) == 10

    def test_iterative_closest_point_basic(self, global_data):
        icp = registration.IterativeClosestPoint()
        icp.run(*global_data)

    def test_iterative_closest_point_point_to_point(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint(max_iteration=300, max_correspondence_distance=0.1)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.1

    def test_iterative_closest_point_point_to_point_with_scaling(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint(with_scaling=True)
        result = icp.run(*global_data, init=ground_truth_world)
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.01

    def test_iterative_closest_point_point_to_plane(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.PLANE,
                                                 max_iteration=100,
                                                 max_correspondence_distance=0.05)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.1

    def test_iterative_closest_point_point_to_plane_with_robust_kernel(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.PLANE,
                                                 max_iteration=100,
                                                 max_correspondence_distance=0.05,
                                                 kernel=registration.KernelTypes.TUKEY)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.1

    def test_iterative_closest_point_color(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.COLOR,
                                                 max_correspondence_distance=0.4)
        result = icp.run(*global_data, init=[0, 0, 0.5])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.1

    def test_iterative_closest_point_point_to_point_multi_scale(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint()
        result = icp.run_multi_scale(*global_data,
                                     init=[0, 0, 0.5],
                                     source_scales=[0.02, 0.01, 0.005],
                                     iterations=[300, 200, 100],
                                     radius_multiplier=[3, 5])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.05

    def test_iterative_closest_point_point_to_plane_multi_scale(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.PLANE)
        result = icp.run_multi_scale(*global_data,
                                     init=[0.1, 0.1, 0.5],
                                     iterations=[200, 100, 50])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.05

    def test_iterative_closest_point_color_multi_scale(self, global_data, ground_truth_world):
        icp = registration.IterativeClosestPoint(estimation_method=registration.ICPTypes.COLOR)
        result = icp.run_multi_scale(*global_data,
                                     init=[0, 0, 0.5],
                                     source_scales=[0.02, 0.01, 0.005])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.05


class TestRANSAC:

    def test_ransac_constructor(self):
        registration.RANSAC()

    def test_ransac_basic(self, global_data):
        ransac = registration.RANSAC()
        ransac.run(*global_data)

    def test_ransac_point_to_point(self, global_data, feature, ground_truth_world):
        source, target = global_data
        source_feature, target_feature = feature
        ransac = registration.RANSAC()
        result = ransac.run(source=source,
                            target=target,
                            source_feature=source_feature,
                            target_feature=target_feature)
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.5

    def test_ransac_point_to_plane(self, global_data, feature, ground_truth_world):
        source, target = global_data
        source_feature, target_feature = feature
        ransac = registration.RANSAC(estimation_method=registration.ICPTypes.PLANE,
                                     max_correspondence_distance=1.0)
        result = ransac.run(source=source,
                            target=target,
                            source_feature=source_feature,
                            target_feature=target_feature)
        assert not np.allclose(result.transformation, np.eye(4))

    def test_ransac_multi_scale(self, global_data, ground_truth_world):
        source, target = global_data
        ransac = registration.RANSAC()
        result = ransac.run_multi_scale(source=source,
                                        target=target,
                                        source_scales=[0.02, 0.01, 0.005],
                                        iterations=[100000, 50000, 20000],
                                        radius_multiplier=[2, 3])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.5


class TestFastGlobalRegistration:

    def test_fgr_constructor(self):
        registration.FastGlobalRegistration()

    def test_fgr_basic(self, global_data):
        fgr = registration.FastGlobalRegistration()
        fgr.run(*global_data)

    def test_fgr(self, global_data, feature, ground_truth_world):
        source, target = global_data
        source_feature, target_feature = feature
        fgr = registration.FastGlobalRegistration()
        result = fgr.run(source=source,
                         target=target,
                         source_feature=source_feature,
                         target_feature=target_feature)
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.5

    def test_fgr_multi_scale(self, global_data, ground_truth_world):
        source, target = global_data
        fgr = registration.FastGlobalRegistration()
        result = fgr.run_multi_scale(source=source,
                                     target=target,
                                     source_scales=[0.02, 0.01, 0.005])
        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.5


class TestPipeline:

    def test_pipeline_mesh(self, source_from_mesh, target_from_mesh, ground_truth_world):
        source = source_from_mesh
        target = target_from_mesh

        source_down = utils.process_point_cloud(point_cloud=source,
                                                downsample=utils.DownsampleTypes.VOXEL,
                                                downsample_factor=0.01,
                                                estimate_normals=True,
                                                recalculate_normals=True)
        _, source_feature = utils.process_point_cloud(point_cloud=source_down, compute_feature=True)

        target_down = utils.process_point_cloud(point_cloud=target,
                                                downsample=utils.DownsampleTypes.VOXEL,
                                                downsample_factor=0.01,
                                                estimate_normals=True)
        _, target_feature = utils.process_point_cloud(point_cloud=target_down, compute_feature=True)

        ransac = registration.RANSAC(max_correspondence_distance=0.01)
        result = ransac.run_n_times(source=source_down,
                                    target=target_down,
                                    source_feature=source_feature,
                                    target_feature=target_feature)

        icp = registration.IterativeClosestPoint(max_iteration=100, max_correspondence_distance=0.01)
        result = icp.run(source=source_down,
                         target=target_down,
                         init=result.transformation)

        assert np.linalg.norm(result.transformation - ground_truth_world) < 0.05
