#!/usr/bin/env python3

import collections.abc as cabc
from dataclasses import dataclass
from functools import wraps
from typing import Any


class ValueKeepingGenerator(object):
    def __init__(self, g):
        self.gen = g
        self.value = None

    def __iter__(self):
        self.value = yield from self.gen
        return self.value  # noqa

    def send(self, x):
        return self.gen.send(x)


def keep_value_generator(f):
    @wraps(f)
    def g(*args, **kwargs):
        return ValueKeepingGenerator(f(*args, **kwargs))

    return g


def _is_seq(obj):
    return isinstance(obj, cabc.Sequence) and not isinstance(obj, str)


def _is_map(obj):
    return isinstance(obj, cabc.Mapping)


@keep_value_generator
def recursive_iterate(
    obj,
    iter_types=None,
    map_check_func=None,
    seq_check_func=None,
    wait_on_send=None,
    yield_name: bool = False,
    _name_prefix: str = "",
):
    """Return an iterable `iter` to allow to access the nested obj (`obj`) linearly.
    The iterator also supports returns a object with the same hierarchy as
    `obj` but stores values passed from `iter.send(x)`. The returned object
    is stored in `iter.value`.
    `iter_types` specifies the type or a tuple of types that will be returned
    during iteration, use `None` to return all objects.
    `map_check_func` adds additional check for if `obj` is a dict, func(x) -> bool.
    `seq_check_func` adds additional check for if `obj` is a list, func(x) -> bool.
    `wait_on_send`: user will call `iter.send(x)` to send data for every element
    if True. The iter will not work properly if the number of elements sent
    do not match. The data sent will be ignored if `wait_on_send` is False.
    `wait_on_send` is None is for backward compatiblity. The iter could accept
    any data sent back except `None`, which will cause an error.
    'yield_name` decides if to yield the name with the object together. If yes,
    the yield values will be ('m1.m2.m3.name', obj), otherwise only `obj` is returned
    """
    ret = obj

    def _get_name_with_prefix(name):
        return ".".join(map(str, filter(lambda x: x != "", [_name_prefix, name])))

    if _is_map(obj) and (map_check_func is None or map_check_func(obj)):
        ret = {}
        for x in obj:
            cur = yield from recursive_iterate(
                obj[x],
                iter_types,
                map_check_func,
                seq_check_func,
                wait_on_send,
                yield_name=yield_name,
                _name_prefix=_get_name_with_prefix(x),
            )
            ret[x] = cur
    elif _is_seq(obj) and (seq_check_func is None or seq_check_func(obj)):
        ret = []
        for idx, x in enumerate(obj):
            cur = yield from recursive_iterate(
                x,
                iter_types,
                map_check_func,
                seq_check_func,
                wait_on_send,
                yield_name=yield_name,
                _name_prefix=_get_name_with_prefix(idx),
            )
            ret.append(cur)
        if isinstance(obj, tuple):
            ret = tuple(ret)
    else:
        is_yield = True
        if iter_types is not None and not isinstance(obj, iter_types):
            is_yield = False
        if is_yield:
            yield_obj = obj if not yield_name else (_name_prefix, obj)
            cret = yield yield_obj
            # data sent back to `ret`, needs a yield for `send()` to pause here as
            #   both 'send()' and `for` advances the generator
            if wait_on_send is True:
                ret = cret
                yield
            elif wait_on_send is False:
                pass
            else:  # for backward compatbility
                assert wait_on_send is None
                # cret is not None means user calls `send`, it could not distingush
                #   the case that user sends None
                if cret is not None:
                    ret = cret
                    yield
    return ret  # noqa


def create_pair(lhs, rhs):
    """Create a pair of objects, handles list/dict automatically
    Could be used with recursive_iterate to match two dicts etc.
    """
    if _is_seq(lhs):
        assert _is_seq(rhs)
        return PairedSeq(lhs, rhs)
    elif _is_map(lhs):
        assert _is_map(rhs)
        return PairedDict(lhs, rhs)
    return Pair(lhs, rhs)


@dataclass
class Pair(object):
    lhs: Any
    rhs: Any

    def to_tuple(self):
        return (self.lhs, self.rhs)


class PairedSeq(cabc.Sequence):
    def __init__(self, lhs, rhs):
        assert _is_seq(lhs)
        assert _is_seq(rhs)
        assert len(lhs) == len(rhs)
        self.lhs = lhs
        self.rhs = rhs

    def __len__(self):
        assert len(self.lhs) == len(self.rhs)
        return len(self.lhs)

    def __getitem__(self, idx):
        return create_pair(self.lhs[idx], self.rhs[idx])


class PairedDict(cabc.Mapping):
    def __init__(self, lhs, rhs):
        assert _is_map(lhs)
        assert _is_map(rhs)
        self.lhs = lhs
        self.rhs = rhs

    def __len__(self):
        assert len(self.lhs) == len(self.rhs)
        return len(self.lhs)

    def __getitem__(self, key):
        return create_pair(self.lhs[key], self.rhs[key])

    def __iter__(self):
        return iter(self.lhs)
