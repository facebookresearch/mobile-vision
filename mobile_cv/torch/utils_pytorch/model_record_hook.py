#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import torch
from mobile_cv.arch.utils.backend_utils import get_cpu_copy
from mobile_cv.common.misc import iter_utils as iu


class ModelRecordHook(object):
    def __init__(self, module_names):
        self.items = []
        self.module_names = module_names
        self._hook_handles = None

    def __call__(self, module, input, output):
        self.items.append(
            (
                self.module_names[id(module)],
                module.__class__.__name__,
                get_cpu_copy(input),
                get_cpu_copy(output),
            )
        )
        return output

    def set_handles(self, handles):
        self._hook_handles = handles

    def remove_handles(self):
        for x in self._hook_handles:
            x.remove()
        self._hook_handles = None

    def get_items_dict(self):
        ret = {x[0]: x for x in self.items}
        return ret


def _compute_tensor_norm(data):
    citer = iu.recursive_iterate(data, iter_types=torch.Tensor, yield_name=True)
    total_ret = 0
    for name, x in citer:
        if x.dtype not in [torch.float, torch.float16]:
            print(f"tensor {name} not float, dtype={x.dtype}, val={x}")
        else:
            total_ret += x.norm()
    return total_ret


def _compute_tensor_diff(data1, data2):
    data = iu.create_pair(data1, data2)
    citer = iu.recursive_iterate(data, iter_types=iu.Pair, yield_name=True)
    total_diff = 0
    for _name, x in citer:
        if isinstance(x.lhs, torch.Tensor):
            total_diff += (x.lhs.float() - x.rhs.float()).norm()
    return total_diff


def print_hook_items(items):
    for idx, (name, class_name, inputs, outputs) in enumerate(items):
        print(
            f"Idx {idx} {class_name}, {name}, "
            f"inputs {_compute_tensor_norm(inputs)}, "
            f"outputs {_compute_tensor_norm(outputs)}."
        )


def get_submodule_names(module: torch.nn.Module):
    ret = {}
    for name, mm in module.named_modules():
        ret[id(mm)] = name

    return ret


def add_model_record_hook(model: torch.nn.Module):
    hook = ModelRecordHook(get_submodule_names(model))
    handles = []

    def reg_func(module):
        handle = module.register_forward_hook(hook)
        handles.append(handle)

    model.apply(reg_func)
    hook.set_handles(handles)
    return hook


def compare_hook_items(model1_hook, model2_hook):
    assert len(model1_hook.items) == len(model2_hook.items)
    ret = []
    for m1_item, m2_item in zip(model1_hook.items, model2_hook.items):
        module_name = (
            m1_item[0] if m1_item[0] == m2_item[0] else f"{m1_item[0]} <=> {m2_item[0]}"
        )
        class_name = (
            m1_item[1] if m1_item[1] == m2_item[1] else f"{m1_item[1]} <=> {m2_item[1]}"
        )
        input_diff = _compute_tensor_diff(m1_item[2], m2_item[2])
        output_diff = _compute_tensor_diff(m1_item[3], m2_item[3])
        ret.append((module_name, class_name, input_diff, output_diff))

        if input_diff > 0 or output_diff > 0:
            yield (module_name, class_name, input_diff, output_diff)


def print_hook_items_difference(model1_hook, model2_hook, max_print_count=None):
    hook_diffs = compare_hook_items(model1_hook, model2_hook)
    counter = 0
    for item in hook_diffs:
        if item[2] > 0 or item[3] > 0:
            print(f"{item[0]}, {item[1]}, input_diff={item[2]}, output_diff={item[3]}")
            counter += 1
            if max_print_count is not None and counter >= max_print_count:
                print(f"Reaching the print limit {max_print_count}")
                break
