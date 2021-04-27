#!/usr/bin/env python3

from functools import lru_cache

from iopath.common.file_io import PathManager


@lru_cache()
def get_path_manager() -> PathManager:
    """target:
    "//mobile-vision/mobile_cv/common:fb",
    """
    path_manager = PathManager()
    return path_manager
