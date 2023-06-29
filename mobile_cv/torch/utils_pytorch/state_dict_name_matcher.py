#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

import torch


def get_state_dict_name_mapping(
    model1_state_dict: Dict[str, torch.Tensor],
    model2_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, str]:
    """Find the best name mapping between two state dicts, based on the name
    similarity and tensor size. It may NOT always give correct results.
    """
    m1_state_dict_shapes = {k: v.shape for k, v in model1_state_dict.items()}
    m2_state_dict_shapes = {k: v.shape for k, v in model2_state_dict.items()}
    assert len(m1_state_dict_shapes) == len(
        m2_state_dict_shapes
    ), f"m1_state_dict_shapes: {m1_state_dict_shapes}, \n m2_state_dict_shapes {m2_state_dict_shapes}"

    ret = get_matching(m1_state_dict_shapes, m2_state_dict_shapes)
    for m1_key, m2_key in ret.items():
        # when the name of the key is not changed, it needs to match the order
        if m1_key in m2_state_dict_shapes:
            assert m1_key == m2_key, f"m1_key: {m1_key}, m2_key: {m2_key}"
        assert (
            m1_state_dict_shapes[m1_key] == m2_state_dict_shapes[m2_key]
        ), f"m1_key: {m1_key}, m2_key: {m2_key}"

    return ret


def levenshtein(s1: str, s2: str):
    """
    Computer distance between two strings
    from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_matching(
    fp32_state_dict_shapes: Dict[str, torch.Size],
    qat_state_dict_shapes: Dict[str, torch.Size],
):
    """
    Find the best mapping between fp32 and qat state_dict items based on the
    name difference and the shape of the tensors.
    The code groups the items based on name prefix/postfix and shape, and only match
    keys for the groups with the same name to improve the matching speed.
    NOTE: This does not guarantee that we could find the right match as there
    could have ambiguities
    """
    assert len(fp32_state_dict_shapes) == len(qat_state_dict_shapes)

    def _dict_to_list(sds):
        return [Item(x, y) for x, y in sds.items()]

    fp32_items = _dict_to_list(fp32_state_dict_shapes)
    qat_items = _dict_to_list(qat_state_dict_shapes)

    # find the best indices to group the keys
    num_parts_first, num_parts_last = find_max_possible_num_groups(
        fp32_items, qat_items
    )

    # divide items to sub groups
    fp32_groups = get_all_groups_from_items(fp32_items, num_parts_first, num_parts_last)
    qat_groups = get_all_groups_from_items(qat_items, num_parts_first, num_parts_last)
    assert (
        fp32_groups.keys() == qat_groups.keys()
    ), f"Mismatched groups, fp32 {fp32_groups.keys()}, qat {qat_groups.keys()}"

    ret = {}
    # match each sub group indivdually
    for group_index in fp32_groups.keys():
        fp32_items = fp32_groups[group_index]
        qat_items = qat_groups[group_index]
        cur_ret = get_match_keys_bipartite(fp32_items, qat_items)
        ret.update(cur_ret)

    return ret


@dataclass
class Item:
    name: str
    shape: torch.Size
    parts: Tuple[str] = field(init=False)

    def __post_init__(self):
        self.parts = tuple(self.name.split("."))

    def name_first(self, num_parts):
        parts = self.parts
        assert len(parts) >= num_parts
        return parts[:num_parts]

    def name_last(self, num_parts):
        if num_parts == 0:
            return ()
        parts = self.parts
        assert len(parts) >= num_parts
        return parts[-num_parts:]


class GroupIndex(NamedTuple):
    """Index used to group different items"""

    prefix: Tuple[str]
    postfix: Tuple[str]
    shape: torch.Size

    def is_match(self, item: Item):
        """If the two items have the same prefix/postfix and shape, then they are
        in the same group
        """
        shape = item.shape
        if self.prefix and self.prefix != item.name_first(len(self.prefix)):
            return False
        if self.postfix and not self.postfix != item.name_last(len(self.postfix)):
            return False
        if self.shape is not None and shape != self.shape:
            return False
        return True


def get_group_index(item: Item, num_parts_first, num_parts_last):
    """Creare grooup index from an item"""
    name_first = item.name_first(num_parts_first)
    name_last = item.name_last(num_parts_last)
    return GroupIndex(name_first, name_last, item.shape)


def get_all_groups_from_items(items: List[Item], num_parts_first, num_parts_last):
    """Divide items into different groups, we only need to match keys for groups
    with the same group_index
    """
    ret = defaultdict(lambda: [])
    for item in items:
        gi = get_group_index(item, num_parts_first, num_parts_last)
        ret[gi].append(item)
    return ret


def find_max_possible_num_groups(fp32_items: List[Item], qat_items: List[Item]):
    num_parts_last = 1

    fp32_min_parts = min(len(x.parts) for x in fp32_items)
    qat_min_parts = min(len(x.parts) for x in qat_items)
    max_first = min(fp32_min_parts, qat_min_parts)
    max_first = max(max_first - 2, 1)

    for num_parts_first in range(max_first, 0, -1):
        fp32_groups = get_all_groups_from_items(
            fp32_items, num_parts_first, num_parts_last
        )
        qat_groups = get_all_groups_from_items(
            qat_items, num_parts_first, num_parts_last
        )
        if len(fp32_groups) == len(qat_groups):
            return num_parts_first, num_parts_last

    return (0, num_parts_last)


def get_match_keys_bipartite(fp32_items: List[Item], qat_items: List[Item]):
    """
    Find the best mapping between fp32 and qat state_dict items based on the
    name difference and the shape of the tensors.
    NOTE: This does not guarantee that we could find the right match as there
    could have ambiguities
    """
    assert len(fp32_items) == len(qat_items)
    dim = len(fp32_items)
    cost = np.zeros((dim, dim))
    for ii in range(dim):
        for jj in range(dim):
            fk, qk = fp32_items[ii], qat_items[jj]
            # string distance
            cost[ii, jj] = levenshtein(fk.name, qk.name)
            # use tensor shape matching as a soft constraint
            if fk.shape != qk.shape:
                cost[ii, jj] += 100000

    from scipy.optimize import linear_sum_assignment  # @manual

    # bipartite matching with minimal cost
    row_ind, col_ind = linear_sum_assignment(cost)
    ret = {fp32_items[ri].name: qat_items[ci].name for ri, ci in zip(row_ind, col_ind)}
    cost_sum = cost[row_ind, col_ind].sum()
    if cost_sum > 0:
        for fp32_key, qat_key in ret.items():
            if fp32_key != qat_key:
                print(f"{fp32_key} -> {qat_key}")
        # print(f"Best matching cost: {cost_sum}")

    return ret


def get_list_mapping(list1: List[str], list2: List[str]) -> Dict[str, str]:
    """Get the best mapping between 2 lists of strings"""
    ret = []

    dim1 = len(list1)
    dim2 = len(list2)

    cost = np.zeros((dim1, dim2))
    for ii in range(dim1):
        for jj in range(dim2):
            cost[ii, jj] = levenshtein(list1[ii], list2[jj])

    from scipy.optimize import linear_sum_assignment  # @manual

    # bipartite matching with minimal cost
    row_ind, col_ind = linear_sum_assignment(cost)
    ret = {list1[ri]: list2[ci] for ri, ci in zip(row_ind, col_ind)}
    cost_sum = cost[row_ind, col_ind].sum()
    if cost_sum > 0:
        for list1_key, list2_key in ret.items():
            if list1_key != list2_key:
                print(f"{list1_key} -> {list2_key}")

    return ret
