#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.py import dynamic_import


def is_oss():
    try:
        import mobile_cv.common.fb.open_source_canary as open_source_canary

        assert open_source_canary
        ret = False
    except ImportError:
        ret = True

    return ret


def fb_overwritable():
    """Decorator on function that has alternative internal implementation"""

    def deco(oss_func):
        if is_oss():
            return oss_func
        else:
            oss_module = oss_func.__module__
            fb_module = oss_module + "_fb"  # xxx.py -> xxx_fb.py
            fb_func = dynamic_import("{}.{}".format(fb_module, oss_func.__name__))
            return fb_func

    return deco
