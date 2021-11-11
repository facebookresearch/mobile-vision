#!/usr/bin/env python3

from functools import lru_cache

from iopath.common.file_io import HTTPURLHandler, PathManager


@lru_cache()
def get_path_manager() -> PathManager:
    """target:
    "//mobile-vision/mobile_cv/mobile_cv/common:fb",
    """
    path_manager = PathManager()
    path_manager.register_handler(HTTPURLHandler())
    return path_manager
