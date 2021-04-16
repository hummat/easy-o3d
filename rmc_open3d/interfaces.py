import copy
import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import open3d as o3d

from .utils import eval_data


class RegistrationInterface(ABC):

    def __init__(self, data_to_cache: dict = None, **kwargs) -> None:

        self.cached_data = dict()
        if data_to_cache is not None:
            self.cache_data(data=data_to_cache, **kwargs)

    def _eval_data(self, data: Any, **kwargs: Any) -> Any:
        try:
            if data in self.cached_data:
                return self.cached_data[data]
        except TypeError:
            hash = hashlib.md5(data)
            if hash in self.cached_data:
                return self.cached_data[hash]
        return eval_data(data=data, **kwargs)

    def cache_data(self, data: dict, **kwargs: Any) -> None:
        for key, value in data.items():
            self.cached_data[key] = eval_data(value, **kwargs)

    def draw_registration_result(self,
                                 source: Any,
                                 target: Any,
                                 pose: np.ndarray = np.eye(4),
                                 draw_coordinate_frames=True,
                                 **kwargs: Any):
        _source = copy.deepcopy(self._eval_data(data=source, **kwargs))
        _target = copy.deepcopy(self._eval_data(data=target, **kwargs))

        _source.paint_uniform_color([0.8, 0, 0])
        _target.paint_uniform_color([0.8, 0.8, 0.8])

        _source.transform(pose)

        to_draw = [_source, _target]
        if draw_coordinate_frames:
            to_draw.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=500))
            to_draw.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=250).transform(pose))

        o3d.visualization.draw_geometries(to_draw)

    @abstractmethod
    def run(self, **kwargs: Any):
        raise NotImplementedError("A derived class should implemented this method.")
