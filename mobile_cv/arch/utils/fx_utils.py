#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import itertools
from collections import defaultdict
from typing import List, Callable, Optional

import mobile_cv.arch.utils.model_utils as mu
import torch
import torch.fx


def is_direct_submodule_node(
    node: torch.fx.Node,
    submodule_name: str,
    input_names: List[str],
    output_names: List[str],
):
    """Check if a node belongs to a sub module of the model, i.e., the name of the
    node starts with the sub module name
    input_names and output_names are the names for the input/output of the sub module
    """
    if node.name in input_names:
        return False
    if node.name.startswith(submodule_name) or node.name in output_names:
        return True
    return False


def get_direct_submodule_nodes(
    model: torch.fx.GraphModule,
    submodule_name: str,
    input_names: List[str],
    output_names: List[str],
):
    direct_sub_nodes = [
        x
        for x in model.graph.nodes
        if is_direct_submodule_node(x, submodule_name, input_names, output_names)
    ]
    return direct_sub_nodes


def get_reference_submodule_node(
    sub_nodes: List[torch.fx.Node], input_names: List[str]
):
    """Get nodes that uses the `sub_nodes`"""
    ret = []
    for node in sub_nodes:
        cur_inputs = node.all_input_nodes
        for cur_input in cur_inputs:
            if cur_input not in sub_nodes and cur_input.name not in input_names:
                ret.append(cur_input)
    return ret


def get_submodule_nodes(
    model: torch.fx.GraphModule,
    submodule_name: str,
    input_names: List[str],
    output_names: List[str],
):
    """Get all the nodes for the sub module with name `submodule_name`,
    `input_names` and `output_names` are the input/output names of the sub module
    """
    direct_sub_nodes = get_direct_submodule_nodes(
        model, submodule_name, input_names, output_names
    )
    ref_nodes = get_reference_submodule_node(direct_sub_nodes, input_names)
    ret = [
        x
        for x in model.graph.nodes
        if x in itertools.chain(direct_sub_nodes, ref_nodes)
    ]
    return ret


def get_input_nodes(nodes: List[torch.fx.Node]):
    """Extract the input nodes from a list of nodes
    Input nodes are the nodes that are not produced by other nodes
    """
    ret = []
    for node in nodes:
        all_inputs = node.all_input_nodes
        for cur_input in all_inputs:
            if cur_input not in nodes:
                ret.append(cur_input)
    assert len(ret) > 0
    return ret


def get_output_nodes(nodes: List[torch.fx.Node]):
    """Extract the output nodes from a list of nodes
    Output nodes are the nodes that are not used by other nodes
    """
    ret = []
    all_inputs = set()
    for node in nodes:
        cur_all_inputs = node.all_input_nodes
        all_inputs.update(set(cur_all_inputs))

    for node in nodes:
        if node not in all_inputs:
            ret.append(node)

    assert len(ret) > 0
    return ret


def get_nodes_from_name(nodes: List[torch.fx.Node], names: List[str]):
    """Get all nodes with the given names"""
    ret = [x for x in nodes if x.name in names]
    return ret


def create_sub_graph(
    sub_nodes: List[torch.fx.Node], input_names: List[str], output_names: List[str]
):
    """Create a graph for the given nodes"""
    graph = torch.fx.Graph()

    # create input
    for cur in input_names:
        graph.placeholder(cur)

    # copy all nodes
    # old to new
    node_map = {}
    node_map_input = {x.name: x for x in graph.nodes}
    for node in sub_nodes:
        node_map[node] = graph.node_copy(
            node,
            lambda n: node_map_input[n.name]
            if n.name in node_map_input
            else node_map[n],
        )

    # create output
    output_nodes = get_nodes_from_name(graph.nodes, output_names)
    for cur in output_nodes:
        graph.output(cur)

    graph.lint()
    return graph


def combine_model_with_subgraph(
    model: torch.fx.GraphModule, sub_name: str, sub_graph: torch.fx.Graph
):
    """Create a new model by combining the sub module `model.sub_name` with
    the graph `sub_graph` so that the new model has a forward function.
    """
    sub_model = copy.deepcopy(getattr(model, sub_name))
    assert sub_model is not None

    # remove sub_name prefix as they will be called inside the sub module,
    # e.g., sub1.layer1.conv -> layer1.conv
    for node in sub_graph.nodes:
        if node.op in ("get_attr", "call_module") and node.target.startswith(
            sub_name + "."
        ):
            node.target = node.target[(len(sub_name) + 1) :]

    # copy attributes in parent module to sub modules so that they could be called
    # from the sub module
    # e.g., model.sub1_attr1 -> model.sub1.sub1_attr1
    for node in sub_graph.nodes:
        if node.op == "get_attr" and hasattr(model, node.target):
            setattr(sub_model, node.target, getattr(model, node.target))

    ret = torch.fx.GraphModule(sub_model, sub_graph)
    ret.graph.lint()
    return ret


def create_sub_model(
    model: torch.fx.GraphModule,
    submodule_name: str,
    input_names: List[str],
    output_names: List[str],
):
    """Extract `model.submodule_name` from `model` as a new model, the new model
    contains the forward function that could be run independently.
    `input_names` and `output_names` are the input/output names of the sub module.
    This is useful for sub modules in `model` that do not have forward functions.
    """
    sub_nodes = get_submodule_nodes(model, submodule_name, input_names, output_names)
    sub_graph = create_sub_graph(sub_nodes, input_names, output_names)
    sub_model = combine_model_with_subgraph(model, submodule_name, sub_graph)
    return sub_model


def replace_sub_model(
    model: torch.fx.GraphModule,
    submodule_name: str,
    input_names: List[str],
    output_names: List[str],
    sub_model_to_replace: torch.nn.Module,
):
    """Replace the sub module `model.submodule_name` with `sub_model_to_replace`
    for the case that `model.submodule_name` does not have a forward function and
    it is called directly inside `model.forward()`
    """
    model = copy.deepcopy(model)
    assert hasattr(model, submodule_name)
    sub_nodes = get_submodule_nodes(model, submodule_name, input_names, output_names)
    input_nodes = get_nodes_from_name(model.graph.nodes, input_names)
    output_nodes = get_nodes_from_name(model.graph.nodes, output_names)
    assert len(output_nodes) == 1, output_nodes
    with model.graph.inserting_before(sub_nodes[0]):
        new_node = model.graph.call_module(submodule_name, args=tuple(input_nodes))
    output_nodes[0].replace_all_uses_with(new_node)
    for node in sub_nodes[::-1]:
        model.graph.erase_node(node)
    setattr(model, submodule_name, sub_model_to_replace)
    model.recompile()
    return model


def extract_submodule_as_model(
    model: torch.fx.GraphModule,
    submodule_name: str,
    input_names: List[str],
    output_names: List[str],
    preserve_attrs: Optional[List[str]] = None,
):
    """Create a new model that is exactly the same as `model` but
    `model.submodule_name` has its own forward function and could be called
    independently
    `input_names` and `output_names` are the input/output names of the sub module.
    preserve_attrs: attributes in model and sub modules to preserve after the
    new model is created
    """
    if preserve_attrs is not None:
        attrs = mu.collect_model_attributes(model, preserve_attrs)
    sub_model = create_sub_model(model, submodule_name, input_names, output_names)
    print(output_names)
    ret = replace_sub_model(model, submodule_name, input_names, output_names, sub_model)
    if preserve_attrs:
        mu.apply_model_attributes(ret, attrs)
    return ret


def find_closest_input_node(node: torch.fx.Node, check_func: Callable):
    queue = [node]
    has_in_queue = {node}
    while len(queue) > 0:
        cur = queue.pop(0)
        if check_func(cur):
            return cur
        for pnode in cur.all_input_nodes:
            if pnode not in has_in_queue:
                queue.append(pnode)
                has_in_queue.add(pnode)
    return None


def get_reference_nodes_map(nodes: List[torch.fx.Node]):
    """Return a map where the value is the list of nodes that uses the key as input"""
    # x: [nodes using x as input]
    ret = defaultdict(set)
    for node in nodes:
        if node not in ret:
            ret[nodes] = set()
        for pnode in node.all_input_nodes:
            ret[pnode].add(node)
    return ret


def find_closest_output_node(node, check_func, reference_nodes_map):
    queue = [node]
    has_in_queue = {node}
    while len(queue) > 0:
        cur = queue.pop(0)
        if check_func(cur):
            return cur
        for nnode in reference_nodes_map[cur]:
            if nnode not in has_in_queue:
                queue.append(nnode)
                has_in_queue.add(nnode)
    return None


def is_quantize_op(node: torch.fx.Node):
    """Check if the node is a node to quantize the tensor"""
    if node.op == "call_function" and node.name.startswith("quantize_per_tensor"):
        return True
    return False


def is_dequant_op(node: torch.fx.Node):
    """Check if the node is a dequantization node"""
    if node.op == "call_method" and node.target == "dequantize":
        return True
    return False


def expand_subnodes_for_quantized_module(
    model: torch.fx.GraphModule, sub_nodes: List[torch.fx.Node]
):
    """Get all the nodes that producing or using `sub_nodes`, while the begin and
    the end of the nodes will be quantization and dequantization nodes
    """
    ret = []
    reference_nodes_map = get_reference_nodes_map(model.graph.nodes)
    queue = sub_nodes[:]
    has_in_queue = set(queue)
    while len(queue) > 0:
        cur = queue.pop(0)
        ret.append(cur)
        if not is_quantize_op(cur) and not is_dequant_op(cur):
            for pnode in cur.all_input_nodes:
                if pnode not in has_in_queue:
                    queue.append(pnode)
                    has_in_queue.add(pnode)
            for nnode in reference_nodes_map[cur]:
                if nnode not in has_in_queue:
                    queue.append(nnode)
                    has_in_queue.add(nnode)

    # add additional nodes needed for quantize_per_tensor
    for node in ret:
        if not is_quantize_op(node):
            continue
        assert len(node.all_input_nodes) == 3
        for idx, pnode in enumerate(node.all_input_nodes):
            if idx == 0:
                continue
            assert pnode.op == "get_attr", pnode
            ret.append(pnode)

    ret = [x for x in model.graph.nodes if x in ret]
    return ret


def get_subnodes_for_quantized_module(model: torch.fx.GraphModule, submodule_name: str):
    """Returns all the nodes that the sub module `model.submodule_name` uses,
    assuming that the sub module starts with quantixation and ends with dequantization
    nodes
    """
    direct_sub_nodes = get_direct_submodule_nodes(model, submodule_name, [], [])
    all_sub_nodes = expand_subnodes_for_quantized_module(model, direct_sub_nodes)
    return all_sub_nodes


def get_inputs_for_quantized_submodule(sub_nodes: List[torch.fx.Node]):
    """Get all the quantization ops"""
    ret = []
    for node in sub_nodes:
        if is_quantize_op(node):
            ret.append(node.all_input_nodes[0])
    return ret


def get_outputs_for_quantized_submodule(sub_nodes: List[torch.fx.Node]):
    """Get all the dequantization ops"""
    ret = []
    for node in sub_nodes:
        if is_dequant_op(node):
            ret.append(node)
    return ret


def extract_quantized_submodule_as_model(
    model: torch.fx.GraphModule,
    submodule_name: str,
    preserve_attrs: Optional[List[str]] = None,
):
    """Create a new model that is exactly the same as `model` but
    `model.submodule_name` has its own forward function and could be called
    independently, assuming that the sub module starts with quantization and ends with
    dequamtization ops
    """
    sub_nodes = get_subnodes_for_quantized_module(model, submodule_name)
    input_nodes = get_inputs_for_quantized_submodule(sub_nodes)
    input_names = [x.name for x in input_nodes]
    output_nodes = get_outputs_for_quantized_submodule(sub_nodes)
    output_names = [x.name for x in output_nodes]
    return extract_submodule_as_model(
        model, submodule_name, input_names, output_names, preserve_attrs=preserve_attrs
    )
