#!/usr/bin/env python3

from iopath.common.file_io import PathManager
from functools import lru_cache


@lru_cache()
def get_path_manager() -> PathManager:
    """target:
    "//mobile-vision/mobile_cv/common:fb",
    """
    path_manager = PathManager()
    return path_manager
