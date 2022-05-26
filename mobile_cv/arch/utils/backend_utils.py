#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from mobile_cv.common.misc import iter_utils as iu


def move_to_device(data, device: str):
    """Move data to the given device, data could be a nested dict/list"""
    diter = iu.recursive_iterate(data, iter_types=torch.Tensor)
    for cur in diter:
        diter.send(cur.to(device))

    return diter.value


class GPUWrapper(torch.nn.Module):
    """A simple wrapper to move the module to run on GPU"""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module.cuda()
        self.training = module.training

    def forward(self, data: torch.Tensor):
        data = data.cuda()
        ret = self.module(data)
        ret = ret.cpu()
        return ret


def seq_module_list_to_gpu(module: torch.nn.ModuleList) -> torch.nn.ModuleList:
    """Move a ModuleList who are being used as follow to GPU:
    for layer in module_list:
        x = layer(x)
    After the call, the above code will still work but the sub modules run on GPU
    """
    assert isinstance(module, torch.nn.ModuleList)
    seq = torch.nn.Sequential(*module)
    seq = GPUWrapper(seq)
    # still return a ModuleList so that the interface matches
    ret = torch.nn.ModuleList([seq])
    return ret
