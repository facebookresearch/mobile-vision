#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import os
from typing import Any, Dict, List, Tuple

import torch
from mobile_cv.common.misc.oss_utils import fb_overwritable
from mobile_cv.common.utils_io import get_path_manager

path_manager = get_path_manager()


@fb_overwritable()
def _get_default_path():
    return "./__debug__"


_DEFAULT_PATH = _get_default_path()


def _get_path(uid: str, name: str, create_path: bool = True):
    out_dir = os.path.join(_DEFAULT_PATH, uid)
    if create_path and not path_manager.exists(out_dir):
        path_manager.mkdirs(out_dir)
    out_path = os.path.join(out_dir, f"{name}.pth")
    return out_path


def save_data(uid: str, name: str, data: Any):
    """Usage:
    from mobile_cv.torch.utils_pytorch.debug_utils import save_data
    save_data(uid, name, data)
    """
    out_file = _get_path(uid, name)
    with path_manager.open(out_file, "wb") as fp:
        torch.save(data, fp)
    return out_file


def load_data(uid: str, name: str, load_on_cpu: bool = False):
    out_file = _get_path(uid, name)
    with path_manager.open(out_file, "rb") as fp:
        load_args = {}
        if load_on_cpu:
            load_args["map_location"] = torch.device("cpu")
        data = torch.load(fp, **load_args)
    return data


def list_all_names(uid: str):
    out_dir = os.path.join(_DEFAULT_PATH, uid)
    ret = path_manager.ls(out_dir)
    ret = [x[: -len(".pth")] for x in ret]
    return ret


def load_all_data(uid: str, load_on_cpu: bool = False):
    ret = {}
    all_names = list_all_names(uid)
    for name in all_names:
        ret[name] = load_data(uid, name, load_on_cpu=load_on_cpu)
    return ret


def compare_data(
    input1: Any, input2: Any, _parent_key: str = ""
) -> Tuple[List[str], List[str]]:
    """Compare two data recursively."""

    mismatched_keys: List[str] = []
    matched_keys: List[str] = []

    if type(input1) != type(input2):
        print(f"{_parent_key} types do not match: {type(input1)} != {type(input2)}")
        print(f"{_parent_key} values: {input1} != {input2}")
        mismatched_keys.append(_parent_key)
    elif isinstance(input1, dict) and isinstance(input2, dict):
        matched_keys, sub_mismatched_keys = _compare_dicts_dicts(
            input1, input2, _parent_key
        )
        mismatched_keys.extend(sub_mismatched_keys)
    elif isinstance(input1, list) and isinstance(input2, list):
        matched_keys, sub_mismatched_keys = _compare_dicts_lists(
            input1, input2, _parent_key
        )
        mismatched_keys.extend(sub_mismatched_keys)
    elif isinstance(input1, torch.Tensor) and isinstance(input2, torch.Tensor):
        if not _compare_tensors(input1, input2):
            print(f"{_parent_key} tensor norm does not match")
            mismatched_keys.append(_parent_key)
        else:
            matched_keys = [_parent_key]
    elif input1 != input2:
        print(f"{_parent_key} value does not match: {input1} != {input2}")
        mismatched_keys.append(_parent_key)
    else:
        matched_keys = [_parent_key]

    if matched_keys:
        print(f"{', '.join(matched_keys)} matched")

    return matched_keys, mismatched_keys


def _compare_dicts_dicts(
    input1: Dict[str, Any], input2: Dict[str, Any], parent_key: str
) -> Tuple[List[str], List[str]]:
    mismatched_keys: List[str] = []
    matched_keys: List[str] = []

    for key in input1.keys() | input2.keys():
        new_parent_key: str = f"{parent_key}.{key}" if parent_key else key

        if key not in input1:
            print(f"{new_parent_key} is extra in input2")
            mismatched_keys.append(new_parent_key)
        elif key not in input2:
            print(f"{new_parent_key} is missing in input2")
            mismatched_keys.append(new_parent_key)
        else:
            sub_matched_keys, sub_mismatched_keys = compare_data(
                input1[key], input2[key], new_parent_key
            )
            matched_keys.extend(sub_matched_keys)
            mismatched_keys.extend(sub_mismatched_keys)

    return matched_keys, mismatched_keys


def _compare_dicts_lists(
    input1: List[Any], input2: List[Any], parent_key: str
) -> Tuple[List[str], List[str]]:
    mismatched_keys: List[str] = []
    matched_keys: List[str] = []

    if len(input1) != len(input2):
        print(f"{parent_key} length does not match: {len(input1)} != {len(input2)}")
        mismatched_keys.append(parent_key)
    else:
        matched_keys = [f"{parent_key}[{i}]" for i in range(len(input1))]

    for i in range(len(input1)):
        sub_matched_keys, sub_mismatched_keys = compare_data(
            input1[i], input2[i], f"{parent_key}[{i}]"
        )
        matched_keys.extend(sub_matched_keys)
        mismatched_keys.extend(sub_mismatched_keys)

    return matched_keys, mismatched_keys


def _compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    if tensor1.shape != tensor2.shape:
        return False

    if tensor1.dtype != tensor2.dtype:
        return False

    if tensor1.dtype == torch.bool:
        return torch.allclose(tensor1.float(), tensor2.float())

    return torch.allclose(tensor1, tensor2)
