#!/usr/bin/env python3

"""
Base task
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple, Union

import torch


class TaskBase(ABC):
    def __init__(self):
        self.model_type_registry = {}
        self.register_model_type("fp32", "get_model")
        self.register_model_type("int8", "get_quantized_model")

    def register_model_type(self, name, func_name):
        self.model_type_registry[name] = func_name
        return self

    def get_model_types(self):
        return self.model_type_registry

    def get_model_by_name(self, name, *args, **kwargs):
        """This code will try to run `self.get_{name}_model(*args, **kwargs)`, or
        use the function name registered by `self.register_model_type()`.
        """
        if not hasattr(self, "model_type_registry"):
            raise Exception("Call `super().__init__()` in the task first.")
        if name in self.model_type_registry:
            func_name = self.model_type_registry[name]
        else:
            # construct func name dynamically
            func_name = f"get_{name}_model"
        func = getattr(self, func_name, None)
        if func is None:
            raise Exception(
                f"Invalid model type name {name}, please implement Task.{func_name}(), or register the model type with task.register_model_type()"  # noqa
            )
        return func(*args, **kwargs)

    @abstractmethod
    def get_model(
        self,
    ) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Dict[Any, Any]]]:
        """Return a pytorch model for the task,
        could be a tuple (model, model_attributes), where model_attibutes is a dict
        and its key/values will become the model's attibutes after scripting.
        """
        pass

    def get_quantizable_model(self, full_model: torch.nn.Module) -> torch.nn.Module:
        """Return a quantizabile pytorch model for the task
        full_model: model from `get_model()`
        """
        model = torch.quantization.QuantWrapper(full_model)
        return model

    @abstractmethod
    def get_dataloader(self) -> Iterable[Any]:
        """Return a data loader for the task"""
        pass
