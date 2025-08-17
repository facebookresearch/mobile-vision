#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from functools import lru_cache

from iopath.common.file_io import HTTPURLHandler, PathManager
from mobile_cv.common.misc.oss_utils import fb_overwritable

logger = logging.getLogger(__name__)


@lru_cache()
def get_path_manager(**kwargs) -> PathManager:
    # FIXME: @fb_overwritable can't be stacked with @lru_cache
    return _get_path_manager(**kwargs)


@fb_overwritable()
def _get_path_manager() -> PathManager:
    path_manager = PathManager()
    path_manager.register_handler(HTTPURLHandler())
    try:
        from iopath.fb.manifold import ManifoldPathHandler

        path_manager.register_handler(ManifoldPathHandler())
    except Exception as e:
        logger.warning(f"Fail to import ManifoldPathHandler: {e}")
    return path_manager
