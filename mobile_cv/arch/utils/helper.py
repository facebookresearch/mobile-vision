#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import copy
import logging
import math


logger = logging.getLogger(__name__)


def py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def get_divisible_by(num, divisible_by, min_val=None):
    ret = int(num)
    if min_val is None:
        min_val = divisible_by
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((py2_round(num / divisible_by) or 1) * divisible_by)
        if ret < 0.95 * num:
            ret += divisible_by
    if ret < min_val:
        ret = min_val
    return ret


def filter_kwargs(func, kwargs, log_skipped=True):
    """Filter kwargs based on signature of `func`
    Return arguments that matches `func`
    """
    import inspect

    sig = inspect.signature(func)

    # if *args or **kwargs in the function, return all arguments
    param_types = {param.kind for param in sig.parameters.values()}
    var_types = {
        inspect.Parameter.VAR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    }
    if len(param_types.intersection(var_types)) > 0:
        return kwargs

    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind
        in [
            param.POSITIONAL_OR_KEYWORD,
            param.KEYWORD_ONLY,
            param.POSITIONAL_ONLY,
        ]
    ]

    if log_skipped:
        skipped_args = [x for x in kwargs.keys() if x not in filter_keys]
        if skipped_args:
            logger.warning(f"Arguments {skipped_args} skipped for op {func.__name__}")

    filtered_dict = {
        filter_key: kwargs[filter_key]
        for filter_key in filter_keys
        if filter_key in kwargs
    }
    return filtered_dict


def filtered_func(func, **additional_args):
    """Wrap `func` to take any input dict, arguments not used by `func` will be
    ignored
    """

    def ret_func(**kwargs):
        all_args = {**kwargs, **additional_args}
        filtered_args = filter_kwargs(func, all_args)
        return func(filtered_args)

    return ret_func


def unify_args(aargs):
    """Return a dict of args"""
    if aargs is None:
        return {}
    if isinstance(aargs, str):
        return {"name": aargs}
    assert isinstance(aargs, dict), f"args {aargs} must be a dict or a str"
    return aargs


def merge_unify_args(*args):
    """Unify and merge the dicts, merge in the order of the list"""
    from collections import ChainMap

    # ChainMap merges the dicts from right to left, so swap the order here
    unified_args = [unify_args(x) for x in reversed(args)]
    ret = dict(ChainMap(*unified_args))
    return ret


def update_dict(dest, src, seq_func=None):
    """Update the dict 'dest' recursively.
    Elements in src could be a callable function with signature
        f(key, curr_dest_val)
    seq_func: function to handle how to process corresponding lists
        seq_func(key, src_val, dest_val) -> new_dest_val
        by default, list will be overrided
    """
    for key, val in src.items():
        if isinstance(val, collections.abc.Mapping):
            # dest[key] could be None in the case of a dict
            cur_dest = dest.get(key, {}) or {}
            assert isinstance(cur_dest, dict), cur_dest
            dest[key] = update_dict(cur_dest, val)
        elif (
            seq_func is not None
            and isinstance(val, collections.abc.Sequence)
            and not isinstance(val, str)
        ):
            cur_dest = dest.get(key, []) or []
            assert isinstance(cur_dest, list), cur_dest
            dest[key] = seq_func(key, val, cur_dest)
        else:
            if callable(val) and key in dest:
                dest[key] = val(key, dest[key])
            else:
                dest[key] = val
    return dest


def update_dict_merge_list(dest, src):
    """Update dict as `update_dict` but merge the two lists together when the values
    are list
    """

    def seq_func(key, val, dest_val):
        assert isinstance(val, list), val
        assert isinstance(dest_val, list), dest_val
        return dest_val + val

    return update_dict(dest, src, seq_func)


def merge(kwargs, **all_args):
    """kwargs will override other arguments"""
    return update_dict(all_args, kwargs)


def get_merged_dict(base, *new_dicts):
    ret = copy.deepcopy(base)
    for x in new_dicts:
        assert isinstance(x, dict)
        update_dict(ret, x)
    return ret


def format_dict_expanding_list_values(dic):
    """
    Formatting a dict into a multi-line string representation of its keys and
    values, if the value is a list, expand every element of that list into a new
    line with "-" in indentation (otherwise use space as indentation).
    Eg. {"aaa": [1, [2, 3]], "bbb": (1, 2, 3)} will become:
    aaa
    - 1
    - [2, 3]
    bbb
      (1, 2, 3)
    """
    dic = copy.deepcopy(dic)
    lines = []
    for k, v in dic.items():
        lines.append(k)
        if isinstance(v, list):
            for elem in v:
                lines.append("- {}".format(elem))
        else:
            lines.append("  {}".format(v))
    return "\n".join(lines)
