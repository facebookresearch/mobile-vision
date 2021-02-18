#!/usr/bin/env python3

"""
Base task
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch


class TaskBase(ABC):
    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """Return a pytorch model for the task"""
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
