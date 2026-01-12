#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import inspect
from typing import Any

from mobile_cv.common.misc.py import dynamic_import

fb_overwritable_funcs = set()


def is_oss():
    try:
        import mobile_cv.common.misc.fb.open_source_canary as open_source_canary

        assert open_source_canary
        ret = False
    except ImportError:
        ret = True

    return ret


def check_for_decorator(func: Any):
    """
    Returns True if given :func (method or class) has fb_overwritable decorator.
    Return False otherwise.
    """
    if not inspect.isfunction(func) and not inspect.isclass(func):
        return False

    global fb_overwritable_funcs
    module_and_name = func.__module__ + "." + func.__name__
    return fb_overwritable_funcs and (module_and_name in fb_overwritable_funcs)


def fb_overwritable():
    """Decorator on function that has alternative internal implementation"""

    def _enforce_decorator(func):
        if not check_for_decorator(func):
            # NOTE: we should create a set of custom exceptions for d2go for easier
            # tracking of various errors.
            raise Exception(
                "Missing @fb_overwritable decorator for {}.{}.".format(
                    func.__module__, func.__name__
                )
            )

    def deco(func):
        global fb_overwritable_funcs

        module_name = func.__module__
        fb_overwritable_funcs.add(module_name + "." + func.__name__)

        # If this is a method that's from "_fb" module then we simply return method
        # as is, no replacement is necessary.
        if module_name.endswith("_fb") or is_oss():
            return func

        fb_module = module_name + "_fb"  # xxx.py -> xxx_fb.py
        fb_func = dynamic_import("{}.{}".format(fb_module, func.__name__))

        _enforce_decorator(fb_func)

        return fb_func

    return deco
