#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import types
from typing import Dict, Generic, ItemsView, KeysView, List, Optional, TypeVar, Union

from mobile_cv.common.misc.py import dynamic_import

logger = logging.getLogger(__name__)


VT = TypeVar("VT")
CLASS_OR_FUNCTION_TYPES = (types.FunctionType, type)  # how to type annotate this?


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

        self._obj_map: Dict[str, Union[VT, str]] = {}

    def _do_register(
        self,
        name: str,
        obj: Union[VT, str],
    ) -> None:
        """
        The obj could be either a function/class or a "module-dot-name" string pointing
        to the function/class. When the obj is stirng, the pointed function/calss will
        be registered lazily.
        """
        if name in self._obj_map and not self._allow_override:
            # allow replacing lazy object with actual object
            if not (
                isinstance(self._obj_map[name], str)
                and isinstance(obj, CLASS_OR_FUNCTION_TYPES)
                and self._obj_map[name] == f"{obj.__module__}.{obj.__qualname__}"
            ):
                raise ValueError(
                    "An object named '{}' was already registered in '{}' registry!"
                    " Existing object ({}) vs new object ({})".format(
                        name, self._name, self._obj_map[name], obj
                    )
                )
        self._obj_map[name] = obj

    def register(
        self,
        name: Optional[str] = None,
        obj: Optional[Union[VT, str]] = None,
    ) -> Union[types.FunctionType, None]:
        """
        Register the given object under the the name or `obj.__name__` if name is None.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                nonlocal name
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            assert not isinstance(
                obj, str
            ), f"Can't lazy-register {obj} without specifying name"
            assert isinstance(
                obj, CLASS_OR_FUNCTION_TYPES
            ), f"Can't infer name for non function/class object: {obj}"
            name = obj.__name__
        self._do_register(name, obj)

    def register_dict(self, mapping: Dict[str, Union[VT, str]]) -> None:
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
        if isinstance(ret, str):
            logger.info(f"Resolving lazy object '{ret}' in '{self._name}' registry ...")
            ret = dynamic_import(ret)
            assert isinstance(ret, CLASS_OR_FUNCTION_TYPES)

        return ret

    def get_names(self) -> List[str]:
        return list(self._obj_map.keys())

    def items(self) -> ItemsView[str, Union[VT, str]]:
        # NOTE: won't resolve lazy object
        return self._obj_map.items()

    def __len__(self) -> int:
        return len(self._obj_map)

    def keys(self) -> KeysView[str]:
        return self._obj_map.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._obj_map

    def __getitem__(self, key: str) -> Union[VT, str]:
        # NOTE: won't resolve lazy object
        return self._obj_map[key]
