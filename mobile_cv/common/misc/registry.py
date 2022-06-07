#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import types
from dataclasses import dataclass
from typing import Dict, Generic, ItemsView, KeysView, List, Optional, TypeVar, Union

from mobile_cv.common.misc.py import dynamic_import

logger = logging.getLogger(__name__)


VT = TypeVar("VT")
CLASS_OR_FUNCTION_TYPES = (types.FunctionType, type)  # how to type annotate this?


@dataclass
class LazyRegisterable:
    """
    A dataclass representing an object that is lazy registered.

    module (str): the name of the module where the registration happens, eg. `a.b.c`.
    name (str or None): the qualified name of the object, eg. `MyClass`. When the object
        doesn't have name (eg. a string), this can be None.
    """

    module: str
    name: Optional[str] = None


# NOTE: we can do `VK = TypeVar("VK")` and `Generic[VK, VT]` if the key can be non-string.
class Registry(Generic[VT]):
    """
    The registry that provides name -> object mapping, to support third-party
      users' custom modules.

    To create a registry (inside detectron2):
        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:
        @BACKBONE_REGISTRY.register("MyBackbone")
        class MyBackbone():
            ...
    Or:
        BACKBONE_REGISTRY.register(name="MyBackbone", obj=MyBackbone)
    """

    def __init__(self, name: str, allow_override: bool = False) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._allow_override: bool = allow_override

        self._obj_map: Dict[str, Union[VT, LazyRegisterable]] = {}

    def _do_register(
        self,
        name: str,
        obj: Union[VT, LazyRegisterable],
    ) -> None:
        """
        The obj could be either a function/class or a "module-dot-name" string pointing
        to the function/class. When the obj is stirng, the pointed function/calss will
        be registered lazily.
        """
        if name in self._obj_map and not self._allow_override:
            orig_obj = self._obj_map[name]
            # allow replacing lazy object with actual object
            if isinstance(orig_obj, LazyRegisterable) and not isinstance(
                obj, LazyRegisterable
            ):
                msg = f"orig obj ({orig_obj}) doesn't match the new obj ({obj})"
                if isinstance(obj, CLASS_OR_FUNCTION_TYPES):
                    assert orig_obj.module == obj.__module__, msg
                    if orig_obj.name is not None:
                        assert orig_obj.name == obj.__name__, msg
            else:
                raise ValueError(
                    "An object named '{}' was already registered in '{}' registry!"
                    " Existing object ({}) vs new object ({})".format(
                        name, self._name, orig_obj, obj
                    )
                )
        self._obj_map[name] = obj

    def _register(
        self,
        name: Optional[str],
        obj: Union[VT, LazyRegisterable],
    ):
        """
        Before calling `_do_register`, resolve the `name` from `obj` in case the `name`
        is not explicity set.
        """
        if name is None:
            assert not isinstance(
                obj, LazyRegisterable
            ), f"Can't lazy-register {obj} without specifying name"
            assert isinstance(
                obj, CLASS_OR_FUNCTION_TYPES
            ), f"Can't infer name for non function/class object: {obj}"
            name = obj.__name__
        return self._do_register(name, obj)

    def register(
        self,
        name: Optional[str] = None,
        obj: Optional[Union[VT, LazyRegisterable]] = None,
    ) -> Union[types.FunctionType, None]:
        """
        Register the given object under the the name or `obj.__name__` if name is None.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                self._register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._register(name, obj)

    def register_dict(self, mapping: Dict[str, Union[VT, LazyRegisterable]]) -> None:
        """
        Register a dict of objects
        """
        assert isinstance(mapping, dict)
        [self.register(name, obj) for name, obj in mapping.items()]

    def get(self, name: str, is_raise: bool = True) -> VT:
        """
        Raise an exception if the key is not found if `is_raise` is True,
            return None otherwise.
        The lazy-registered object will be resolved.
        """
        ret = self._obj_map.get(name)

        if ret is None and is_raise:
            raise KeyError(
                "No object named '{}' found in '{}' registry! Available names: {}".format(
                    name, self._name, list(self._obj_map.keys())
                )
            )

        # resolve lazy registration by dynamic importing the object, note that if the
        # module is imported for the first time, it will re-register the actual object
        # thus updating self._obj_map.
        if isinstance(ret, LazyRegisterable):
            logger.info(f"Resolving lazy object '{ret}' in '{self._name}' registry ...")
            if ret.name is None:
                dynamic_import(ret.module)
                ret = self._obj_map.get(name)
            else:
                ret = dynamic_import(f"{ret.module}.{ret.name}")
                assert isinstance(ret, CLASS_OR_FUNCTION_TYPES)

        return ret

    def get_names(self) -> List[str]:
        return list(self._obj_map.keys())

    def items(self) -> ItemsView[str, Union[VT, LazyRegisterable]]:
        # NOTE: won't resolve lazy object
        return self._obj_map.items()

    def __len__(self) -> int:
        return len(self._obj_map)

    def keys(self) -> KeysView[str]:
        return self._obj_map.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._obj_map

    def __getitem__(self, key: str) -> Union[VT, LazyRegisterable]:
        # NOTE: won't resolve lazy object
        return self._obj_map[key]
