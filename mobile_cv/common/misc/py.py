#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path  # NOQA
def import_file(module_name, file_path, make_importable=False):
    import importlib
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def dynamic_import(obj_full_name):
    """
    Dynamically import an object (class or function or global variable).

    Args:
        obj_full_name: full name of the object, eg. ExampleClass/example_foo is defined
        inside a/b.py, the obj_full_name is "a.b.ExampleClass" or "a.b.example_foo".
    Returns:
        The imported object.
    """
    import pydoc

    ret = pydoc.locate(obj_full_name)
    if ret is None:
        raise ImportError("Cannot dynamically locate {}".format(obj_full_name))
    return ret
