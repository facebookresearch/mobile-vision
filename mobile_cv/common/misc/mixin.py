#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Any, Dict, Optional


def dynamic_mixin(
    obj: object,
    new_class: type,
    init_new_class: bool = True,
    init_dict: Optional[Dict[str, Any]] = None,
):
    """
    Dynamically mixin a class to an instantiated object

    If init_new_class, assume that the new_class has a method called
    "dynamic_mixin_init" and run this using init_dict

    Save original model class in object so we can remove the new class
    with dynamic mixin again later
    Note: https://stackoverflow.com/questions/8544983/dynamically-mixin-a-base-class-to-an-instance-in-python
    """
    original_model_class = obj.__class__
    obj.__class__ = type(
        f"{original_model_class.__name__}_{new_class.__name__}",
        (new_class, original_model_class),
        {},
    )

    if init_new_class:
        assert hasattr(new_class, "dynamic_mixin_init")
        if init_dict is not None:
            obj.dynamic_mixin_init(**init_dict)
        else:
            obj.dynamic_mixin_init()

    assert not hasattr(
        obj, "_original_model_class"
    ), f"Dynamic mixin attempting to override original_model_class that already exists: {obj.original_model_class}"
    obj._original_model_class = original_model_class


def remove_dynamic_mixin(obj: object, call_object_remove: bool = True):
    """
    Remove a class added previously by dynamix mixin

    Assumes that the object itself retains a property called _original_model_class.
    Uses the object method remove_dynamic_mixin if call_object_remove specified
    """
    assert hasattr(
        obj, "_original_model_class"
    ), "Unable to remove mixed in class without original_model_class"
    if call_object_remove:
        obj.remove_dynamic_mixin()

    obj.__class__ = type(
        obj._original_model_class.__name__, (obj._original_model_class,), {}
    )
    del obj._original_model_class
