#!/usr/bin/env python3

import torch
import torch.nn as nn


class MixModule(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.ops = nn.ModuleList(modules)
        self.register_buffer("weights", torch.zeros(len(modules)))
        self.weights[0] = 1.0
        # self.output_depth = self.ops[0].output_depth
        # for x in modules:
        #     assert self.output_depth == x.output_depth

    def get_path_count(self):
        return len(self.ops)

    def set_random_path(self):
        path_count = self.get_path_count()
        self.weights.zero_()
        val = torch.randint(0, path_count, (1,), dtype=torch.int32).item()
        self.weights[val] = 1.0

    def forward(self, x):
        y = 0
        for w, op in zip(self.weights, self.ops):
            y += op(x) * w

        return y


def create_mix_blocks(
    block_names, primitives, c_in, c_out, expansion, stride, **kwargs
):
    blocks = []
    for name in block_names:
        cb = primitives[name](c_in, c_out, expansion, stride, **kwargs)
        blocks.append(cb)
    ret = MixModule(blocks)
    ret.output_depth = c_out
    return ret
