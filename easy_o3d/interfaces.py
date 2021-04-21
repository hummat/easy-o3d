import copy
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import open3d as o3d

from .utils import InputTypes, eval_data

RegistrationResult = o3d.pipelines.registration.RegistrationResult
TriangleMesh = o3d.geometry.TriangleMesh
PointCloud = o3d.geometry.PointCloud

logger = logging.getLogger(__name__)


class RegistrationInterface(ABC):

    def __init__(self, data_to_cache: dict = None, **kwargs: Any) -> None:

        self.cached_data = dict()
        if data_to_cache is not None:
            self.add_to_cache(data=data_to_cache, **kwargs)

    def _eval_data(self, data: InputTypes, **kwargs: Any) -> PointCloud:
        try:
            if data in self.cached_data:
                logger.debug(f"Using cached data {data}.")
                return self.cached_data[data]
        except TypeError:
            hash = hashlib.md5(data)
            if hash in self.cached_data:
                return self.cached_data[hash]
        logger.debug(f"Couldn't find data {data} in cashe. Re-evaluating.")
        return eval_data(data=data, **kwargs)

    def add_to_cache(self, data: dict, replace: bool = False, **kwargs: Any) -> None:
        for key, value in data.items():
            try:
                if key not in self.cached_data or replace:
                    logger.debug(f"Adding data with key {key} to cache.")
                    self.cached_data[key] = eval_data(value, **kwargs)
            except TypeError:
                logger.warning(f"Couldn't add data with unhashable key {key}.")

    def replace_in_cache(self, data: dict, **kwargs: Any):
        for key, value in data.items():
            try:
                if key in self.cached_data:
                    logger.debug(f"Replacing data with key {key} in cache.")
                    self.cached_data[key] = eval_data(value, **kwargs)
            except TypeError:
                logger.warning(f"Couldn't raplace data with unhashable key {key}.")

    def get_cache(self):
        return self.cached_data

    def get_from_cache(self, data: InputTypes):
        return self.cached_data[data]

    def draw_registration_result(self,
                                 source: InputTypes,
                                 target: InputTypes,
                                 pose: np.ndarray = np.eye(4),
                                 draw_coordinate_frames: bool = True,
                                 draw_bounding_boxes: bool = False,
                                 overwrite_colors: bool = False,
                                 **kwargs: Any) -> None:
        _source = copy.deepcopy(self._eval_data(data=source, **kwargs))
        _target = copy.deepcopy(self._eval_data(data=target, **kwargs))

        if not _source.has_colors() or overwrite_colors:
            _source.paint_uniform_color([0.8, 0, 0])
        if not _target.has_colors() or overwrite_colors:
            _target.paint_uniform_color([0.8, 0.8, 0.8])

        _source.transform(pose)

        to_draw = [_source, _target]
        if draw_coordinate_frames:
            size = 0.5 * (np.asarray(_source.get_max_bound()) - np.asarray(_source.get_min_bound())).max()
            to_draw.append(TriangleMesh.create_coordinate_frame(size=2 * size))
            to_draw.append(TriangleMesh.create_coordinate_frame(size=size).transform(pose))

        if draw_bounding_boxes:
            to_draw.append(_source.get_axis_aligned_bounding_box())
            to_draw.append(_target.get_axis_aligned_bounding_box())

        o3d.visualization.draw_geometries(to_draw)

    @abstractmethod
    def run(self, source: InputTypes, target: InputTypes, **kwargs: Any) -> RegistrationResult:
        raise NotImplementedError("A derived class should implemented this method.")
