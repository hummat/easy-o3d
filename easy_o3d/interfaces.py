"""Intefaces and base classes.

Classes:
    interfaces.RegistrationInterface: Interface for all registration classes.
"""

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
    """Interface for registration subclasses. Handles data caching and visualization.

    Attributes:
        cached_data (dict): A dictionary for caching data used during registration.        

    Methods:
        add_to_cache(data, replace): Adds `data.values()` to cache, replacing data with existing
                                     dict keys if `replace` is set to `True`.
        replace_in_cache(data): Replaces `data.values()` for existing keys.
        get_cache(): Returns `cached_data`.
        get_from_cache(data_key): Returns value for `data_key` from cache.
        draw_registration_result(source, target, pose, ...): Visualizes the registration result
                                                             of `source` being aligned with `target`
                                                             using `pose`.
    """

    def __init__(self, data_to_cache: dict = None, **kwargs: Any) -> None:
        """
        Args:
            data_to_cache (dict): The data to be cached.
            kwargs (Any): Optional additional keyword arguments used downstream.
        """

        self.cached_data = dict()
        if data_to_cache is not None:
            self.add_to_cache(data=data_to_cache, **kwargs)

    def _eval_data(self, data_key_or_value: InputTypes, **kwargs: Any) -> PointCloud:
        """Returns cached data if possible. Processes input to return point cloud otherwise.

        Args:
            data_key_or_value (InputTypes): data_key_or_value
            kwargs (Any): Optional additional keyword arguments used downstream.

        Returns:
            PointCloud: The cached or loaded/processed point cloud data.
        """

        try:
            if data_key_or_value in self.cached_data:
                logger.debug(f"Using cached data {data_key_or_value}.")
                return self.cached_data[data_key_or_value]
        except TypeError:
            hash = hashlib.md5(data_key_or_value)
            if hash in self.cached_data:
                return self.cached_data[hash]
        logger.debug(f"Couldn't find data {data_key_or_value} in cashe. Re-evaluating.")
        return eval_data(data=data_key_or_value, **kwargs)

    def add_to_cache(self, data: dict, replace: bool = False, **kwargs: Any) -> None:
        """Adds `data.values()` to cache, replacing data with existing dict keys if `replace` is set to `True`.

        Args:
            data (dict): The data to be cached.
            replace (bool): Overwrite existing data in cache.
            kwargs (Any): Optional additional keyword arguments used downstream.
        """

        for key, value in data.items():
            try:
                if key not in self.cached_data or replace:
                    logger.debug(f"Adding data with key {key} to cache.")
                    self.cached_data[key] = eval_data(data=value, **kwargs)
            except TypeError:
                logger.warning(f"Couldn't add data with unhashable key {key}.")

    def replace_in_cache(self, data: dict, **kwargs: Any) -> None:
        """Replaces `data.values()` for existing keys.

        Args:
            data (dict): The data to be replaced.
            kwargs (Any): Optional additional keyword arguments used downstream.
        """

        for key, value in data.items():
            try:
                if key in self.cached_data:
                    logger.debug(f"Replacing data with key {key} in cache.")
                    self.cached_data[key] = eval_data(value, **kwargs)
            except TypeError:
                logger.warning(f"Couldn't raplace data with unhashable key {key}.")

    def get_cache(self) -> dict:
        """Returns `cached_data`."""
        return self.cached_data

    def get_from_cache(self, data_key: InputTypes) -> PointCloud:
        """Returns value for `data_key` from cache.

        Args:
            data_key (InputTypes): Key storing `data.value` in cache.

        Returns:
            PointCloud: The cached point cloud data.
        """
        return self.cached_data[data_key]

    def draw_registration_result(self,
                                 source: InputTypes,
                                 target: InputTypes,
                                 pose: np.ndarray = np.eye(4),
                                 draw_coordinate_frames: bool = True,
                                 draw_bounding_boxes: bool = False,
                                 overwrite_colors: bool = False,
                                 **kwargs: Any) -> None:
        """Visualizes the registration result of `source` being aligned with `target` using `pose`.

        Args:
            source (InputTypes): The source data.
            target (InputTypes): The target data.
            pose (np.ndarray): The 4x4 transformation matrix between `source` and `target`.
            draw_coordinate_frames (bool): Draws coordinate frames for `source` and `target`.
            draw_bounding_boxes (bool): Draws axis-aligned bounding boxes for `source` and `target`.
            overwrite_colors (bool): Overwrites `source` and `target` colors for clearer visualization.
            kwargs (Any): Optional additional keyword arguments used downstream.
        """

        _source = copy.deepcopy(self._eval_data(data_key_or_value=source, **kwargs))
        _target = copy.deepcopy(self._eval_data(data_key_or_value=target, **kwargs))

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
        """Runs the registration algorithm of the derived class.

        Args:
            source (InputTypes): The source data.
            target (InputTypes): The target data.
            kwargs (Any): Optional additional keyword arguments.

        Raises:
            NotImplementedError: A derived class should implemented this method.

        Returns:
            RegistrationResult: The registration result containing relative fitness and RMSE
                                as well as the the transformation between `source` and `target`.
        """
        raise NotImplementedError("A derived class should implemented this method.")
