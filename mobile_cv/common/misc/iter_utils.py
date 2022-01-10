#!/usr/bin/env python3

import collections.abc as cabc
from dataclasses import dataclass
from functools import wraps
from typing import List, Callable, Tuple, Union, Optional, Any


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


def is_seq(obj, strict=False):
    if strict:
        return type(obj) == list or type(obj) == tuple
    # this check is general that NamedTuple etc. will return True
    return isinstance(obj, cabc.Sequence) and not isinstance(obj, str)


def is_map(obj, strict=False):
    if strict:
        return type(obj) == dict
    return isinstance(obj, cabc.Mapping)


def is_container(obj, strict=False):
    return is_seq(obj, strict) or is_map(obj, strict)


def _yield_obj(obj, wait_on_send: bool, yield_name: bool, name_prefix: str):
    ret = obj
    yield_obj = obj if not yield_name else (name_prefix, obj)
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


def recursive_iterate(
    obj: Any,
    iter_types: Optional[Union[Any, Tuple[Any]]] = None,
    map_check_func: Optional[Callable[[Any], bool]] = None,
    seq_check_func: Optional[Callable[[Any], bool]] = None,
    wait_on_send: Optional[bool] = None,
    yield_name: bool = False,
    yield_container: bool = False,
):
    """Return an iterable `iter` to allow access to the nested obj (`obj`) linearly.
    The iterator also supports returns a object with the same hierarchy as
    `obj` but stores values passed from `iter.send(x)`. The returned object
    is stored in `iter.value`.
    `iter_types` specifies the type or a tuple of types that will be returned
    during iteration, use `None` to return all objects.
    `map_check_func` function to check for if `obj` is a dict, func(x) -> bool.
    `seq_check_func` function to check for if `obj` is a list, func(x) -> bool.
    `wait_on_send`: user will call `iter.send(x)` to send data for every element
    if True. The iter will not work properly if the number of elements sent
    do not match. The data sent will be ignored if `wait_on_send` is False.
    `wait_on_send` is None for backward compatibility. The iter could accept
    any data sent back except `None`, which will cause an error.
    'yield_name` decides if to yield the name with the object together. If yes,
    the yield values will be ('m1.m2.m3.name', obj), otherwise only `obj` is
    returned.
    `yield_container` indicates whether to yield the container object (list/dict etc.)
    itself during the iteration. If True, it will be yielded after all elements
    inside it are yielded. At the time a container object is yielded, `iter.sent_value()`
    contains the corresponding container object with the elements inside it replaced
    with the values users have sent. This is useful to process and send the container
    object back to construct the output.
    """
    out_sent_container = []
    ret = _recursive_iterate(
        obj,
        iter_types=iter_types,
        map_check_func=map_check_func,
        seq_check_func=seq_check_func,
        wait_on_send=wait_on_send,
        yield_name=yield_name,
        yield_container=yield_container,
        out_sent_container=out_sent_container,
        _name_prefix="",
    )
    ret.sent_value = lambda: out_sent_container[-1]
    return ret


@keep_value_generator
def _recursive_iterate(
    obj: Any,
    iter_types: Optional[Union[Any, Tuple[Any]]] = None,
    map_check_func: Optional[Callable[[Any], bool]] = None,
    seq_check_func: Optional[Callable[[Any], bool]] = None,
    wait_on_send: Optional[bool] = None,
    yield_name: bool = False,
    yield_container: bool = False,
    out_sent_container: Optional[List[Any]] = None,
    _name_prefix: str = "",
):
    """Return an iterable `iter` to allow access to the nested obj (`obj`) linearly.
    The iterator also supports returns a object with the same hierarchy as
    `obj` but stores values passed from `iter.send(x)`. The returned object
    is stored in `iter.value`.
    `iter_types` specifies the type or a tuple of types that will be returned
    during iteration, use `None` to return all objects.
    `map_check_func` function to check for if `obj` is a dict, func(x) -> bool.
    `seq_check_func` function to check for if `obj` is a list, func(x) -> bool.
    `wait_on_send`: user will call `iter.send(x)` to send data for every element
    if True. The iter will not work properly if the number of elements sent
    do not match. The data sent will be ignored if `wait_on_send` is False.
    `wait_on_send` is None for backward compatibility. The iter could accept
    any data sent back except `None`, which will cause an error.
    'yield_name` decides if to yield the name with the object together. If yes,
    the yield values will be ('m1.m2.m3.name', obj), otherwise only `obj` is returned
    `yield_container` indicates whether to yield the container object (list/dict etc.)
    itself during the iteration. If True, it will be yielded after all elements
    inside it are yielded.
    `out_sent_container` is an empty list provided from the user and use to store
    the container value created based on the values from `send()` during the iteration.
    At the time a container object is yielded, out_sent_container[-1] stores the
    corresponding container object but with the elements inside it replaced with
    the values users have sent.
    """

    def _get_name_with_prefix(name):
        return ".".join(map(str, filter(lambda x: x != "", [_name_prefix, name])))

    if map_check_func is None:
        map_check_func = is_map
    if seq_check_func is None:
        seq_check_func = is_seq

    is_obj_map = map_check_func(obj)
    is_obj_seq = seq_check_func(obj)
    is_obj_container = is_obj_map or is_obj_seq

    # by default yield every object except dict and list
    is_obj_yield = not is_obj_container if not yield_container else True
    if iter_types is not None:
        is_obj_yield = is_obj_yield and isinstance(obj, iter_types)

    ret = obj
    if is_obj_map:
        ret = {}
        for x in obj:
            cur = yield from _recursive_iterate(
                obj[x],
                iter_types,
                map_check_func,
                seq_check_func,
                wait_on_send,
                yield_name=yield_name,
                yield_container=yield_container,
                out_sent_container=out_sent_container,
                _name_prefix=_get_name_with_prefix(x),
            )
            ret[x] = cur
    elif is_obj_seq:
        ret = []
        for idx, x in enumerate(obj):
            cur = yield from _recursive_iterate(
                x,
                iter_types,
                map_check_func,
                seq_check_func,
                wait_on_send,
                yield_name=yield_name,
                yield_container=yield_container,
                out_sent_container=out_sent_container,
                _name_prefix=_get_name_with_prefix(idx),
            )
            ret.append(cur)
        if isinstance(obj, tuple):
            ret = tuple(ret)

    if is_obj_yield:
        # store the sent container in `out_sent_container` to be used outside
        if out_sent_container is not None:
            assert isinstance(out_sent_container, list)
            out_sent_container.append(ret)
        ret = yield from _yield_obj(
            obj,
            wait_on_send=wait_on_send,
            yield_name=yield_name,
            name_prefix=_name_prefix,
        )
        if out_sent_container is not None:
            out_sent_container.pop(-1)

    return ret  # noqa


def create_pair(lhs, rhs):
    """Create a pair of objects, handles list/dict automatically
    Could be used with recursive_iterate to match two dicts etc.
    """
    if is_seq(lhs):
        assert is_seq(rhs)
        return PairedSeq(lhs, rhs)
    elif is_map(lhs):
        assert is_map(rhs)
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
        assert is_seq(lhs)
        assert is_seq(rhs)
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
        assert is_map(lhs)
        assert is_map(rhs)
        self.lhs = lhs
        self.rhs = rhs

    def __len__(self):
        assert len(self.lhs) == len(self.rhs)
        return len(self.lhs)

    def __getitem__(self, key):
        return create_pair(self.lhs[key], self.rhs[key])

    def __iter__(self):
        return iter(self.lhs)
