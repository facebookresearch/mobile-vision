#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from functools import lru_cache

from iopath.common.file_io import HTTPURLHandler, PathManager
from mobile_cv.common.misc.oss_utils import fb_overwritable


@lru_cache()
def get_path_manager() -> PathManager:
    # FIXME: @fb_overwritable can't be stacked with @lru_cache
    return _get_path_manager()


@fb_overwritable()
def _get_path_manager() -> PathManager:
    path_manager = PathManager()
    path_manager.register_handler(HTTPURLHandler())
    return path_manager
