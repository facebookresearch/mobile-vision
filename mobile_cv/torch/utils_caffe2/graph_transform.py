#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""
This module provides reusable helper functions for doing caffe2 graph
    transform.
"""

import collections
import copy
import logging
from typing import Dict, List, Tuple

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from mobile_cv.torch.utils_caffe2.protobuf import get_consumer_map, get_producer_map


logger = logging.getLogger(__name__)


class IllegalGraphTransformError(ValueError):
    """When a graph transform function call can't be executed."""


def _rename_versioned_blob_in_proto(
    proto: caffe2_pb2.NetDef,
    old_name: str,
    new_name: str,
    version: int,
    ssa: List[Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]],
    start_versions: Dict[str, int],
    end_versions: Dict[str, int],
):
    """In given proto, rename all blobs with matched version"""
    # Operater list
    for op, i_th_ssa in zip(proto.op, ssa):
        versioned_inputs, versioned_outputs = i_th_ssa
        for i in range(len(op.input)):
            if versioned_inputs[i] == (old_name, version):
                op.input[i] = new_name
        for i in range(len(op.output)):
            if versioned_outputs[i] == (old_name, version):
                op.output[i] = new_name
    # external_input
    if start_versions.get(old_name, 0) == version:
        for i in range(len(proto.external_input)):
            if proto.external_input[i] == old_name:
                proto.external_input[i] = new_name
    # external_output
    if end_versions.get(old_name, 0) == version:
        for i in range(len(proto.external_output)):
            if proto.external_output[i] == old_name:
                proto.external_output[i] = new_name


def rename_op_input(
    predict_net: caffe2_pb2.NetDef,
    init_net: caffe2_pb2.NetDef,
    op_id: int,
    input_id: int,
    new_name: str,
    from_producer: bool = False,
):
    """
    Rename the op_id-th operator in predict_net, change it's input_id-th input's
        name to the new_name. It also does automatic re-route and change
        external_input and init_net if necessary.
    - It requires the input is only consumed by this op.
    - This function modifies predict_net and init_net in-place.
    - When from_producer is enable, this also updates other operators that consumes
        the same input. Be cautious because may trigger unintended behaviour.
    """
    assert isinstance(predict_net, caffe2_pb2.NetDef)
    assert isinstance(init_net, caffe2_pb2.NetDef)

    init_net_ssa, init_net_versions = core.get_ssa(init_net)
    predict_net_ssa, predict_net_versions = core.get_ssa(
        predict_net, copy.deepcopy(init_net_versions)
    )

    versioned_inputs, versioned_outputs = predict_net_ssa[op_id]
    old_name, version = versioned_inputs[input_id]

    if from_producer:
        producer_map = get_producer_map(predict_net_ssa)
        if not (old_name, version) in producer_map:
            raise NotImplementedError(
                "Can't find producer, the input {} is probably from"
                " init_net, this is not supported yet.".format(old_name)
            )
        producer = producer_map[(old_name, version)]
        rename_op_output(predict_net, producer[0], producer[1], new_name)
        return

    def contain_targets(op_ssa):
        return (old_name, version) in op_ssa[0]

    is_consumer = [contain_targets(op_ssa) for op_ssa in predict_net_ssa]
    if sum(is_consumer) > 1:
        raise IllegalGraphTransformError(
            (
                "Input '{}' of operator(#{}) are consumed by other ops, please use"
                + " rename_op_output on the producer instead. Offending op: \n{}"
            ).format(old_name, op_id, predict_net.op[op_id])
        )

    # update init_net
    _rename_versioned_blob_in_proto(
        init_net, old_name, new_name, version, init_net_ssa, {}, init_net_versions
    )
    # update predict_net
    _rename_versioned_blob_in_proto(
        predict_net,
        old_name,
        new_name,
        version,
        predict_net_ssa,
        init_net_versions,
        predict_net_versions,
    )


def rename_op_output(
    predict_net: caffe2_pb2.NetDef,
    op_id: int,
    output_id: int,
    new_name: str,
):
    """
    Rename the op_id-th operator in predict_net, change it's output_id-th input's
        name to the new_name. It also does automatic re-route and change
        external_output and if necessary.
    - It allows multiple consumers of its output.
    - This function modifies predict_net in-place, doesn't need init_net.
    """
    assert isinstance(predict_net, caffe2_pb2.NetDef)

    ssa, blob_versions = core.get_ssa(predict_net)

    versioned_inputs, versioned_outputs = ssa[op_id]
    old_name, version = versioned_outputs[output_id]

    # update predict_net
    _rename_versioned_blob_in_proto(
        predict_net,
        old_name,
        new_name,
        version,
        ssa,
        {},
        blob_versions,
    )


def get_sub_graph_external_input_output(
    predict_net: caffe2_pb2.NetDef,
    sub_graph_op_indices: List[int],
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Return the list of external input/output of sub-graph,
    each element is tuple of the name and corresponding version in predict_net.

    external input/output is defined the same way as caffe2 NetDef.
    """
    ssa, versions = core.get_ssa(predict_net)

    all_inputs = []
    all_outputs = []
    for op_id in sub_graph_op_indices:
        all_inputs += [inp for inp in ssa[op_id][0] if inp not in all_inputs]
        all_outputs += list(ssa[op_id][1])  # ssa output won't repeat

    # for versioned blobs, external inputs are just those blob in all_inputs
    # but not in all_outputs
    ext_inputs = [inp for inp in all_inputs if inp not in all_outputs]

    # external outputs are essentially outputs of this subgraph that are used
    # outside of this sub-graph (including predict_net.external_output)
    all_other_inputs = sum(
        (ssa[i][0] for i in range(len(ssa)) if i not in sub_graph_op_indices),
        [(outp, versions[outp]) for outp in predict_net.external_output],
    )
    ext_outputs = [outp for outp in all_outputs if outp in set(all_other_inputs)]

    return ext_inputs, ext_outputs


class DiGraph:
    """A DAG representation of caffe2 graph, each vertice is a versioned blob."""

    def __init__(self):
        self.vertices = set()
        self.graph = collections.defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)

    # grab from https://www.geeksforgeeks.org/find-paths-given-source-destination/
    def get_all_paths(self, s, d):
        visited = {k: False for k in self.vertices}
        path = []
        all_paths = []

        def _get_all_paths_util(graph, u, d, visited, path):
            visited[u] = True
            path.append(u)
            if u == d:
                all_paths.append(copy.deepcopy(path))
            else:
                for i in graph[u]:
                    if not visited[i]:
                        _get_all_paths_util(graph, i, d, visited, path)
            path.pop()
            visited[u] = False

        _get_all_paths_util(self.graph, s, d, visited, path)
        return all_paths

    @staticmethod
    def from_ssa(ssa):
        graph = DiGraph()
        for op_id in range(len(ssa)):
            for inp in ssa[op_id][0]:
                for outp in ssa[op_id][1]:
                    graph.add_edge(inp, outp)
        return graph


def _get_dependency_chain(ssa, versioned_target, versioned_source):
    """
    Return the index list of relevant operator to produce target blob from source blob,
        if there's no dependency, return empty list.
    """

    # finding all paths between nodes can be O(N!), thus we can only search
    # in the subgraph using the op starting from the first consumer of source blob
    # to the producer of the target blob.
    consumer_map = get_consumer_map(ssa)
    producer_map = get_producer_map(ssa)
    start_op = min(x[0] for x in consumer_map[versioned_source]) - 15
    end_op = (
        producer_map[versioned_target][0] + 15
        if versioned_target in producer_map
        else start_op
    )
    sub_graph_ssa = ssa[start_op : end_op + 1]
    if len(sub_graph_ssa) > 30:
        logger.warning(
            "Subgraph bebetween {} and {} is large (from op#{} to op#{}), it"
            " might take non-trival time to find all paths between them.".format(
                versioned_source, versioned_target, start_op, end_op
            )
        )

    dag = DiGraph.from_ssa(sub_graph_ssa)
    paths = dag.get_all_paths(versioned_source, versioned_target)  # include two ends
    ops_in_paths = [[producer_map[blob][0] for blob in path[1:]] for path in paths]
    return sorted(set().union(*[set(ops) for ops in ops_in_paths]))


def identify_reshape_sub_graph(
    predict_net: caffe2_pb2.NetDef,
) -> List[List[int]]:
    """
    Idenfity the reshape sub-graph in a protobuf.
    The reshape sub-graph is defined as matching the following pattern:

    (input_blob) -> Op_1 -> ... -> Op_N -> (new_shape) -─┐
        └-------------------------------------------> Reshape -> (output_blob)

    Return:
        List of sub-graphs, each sub-graph is represented as a list of indices
        of the relavent ops, [Op_1, Op_2, ..., Op_N, Reshape]
    """

    ssa, _ = core.get_ssa(predict_net)

    ret = []
    for i, op in enumerate(predict_net.op):
        if op.type == "Reshape":
            assert len(op.input) == 2
            input_ssa = ssa[i][0]
            data_source = input_ssa[0]
            shape_source = input_ssa[1]
            op_indices = _get_dependency_chain(ssa, shape_source, data_source)
            ret.append(op_indices + [i])
    return ret


def remove_reshape_for_fc(predict_net, params):
    """
    In PyTorch nn.Linear has to take 2D tensor, this often leads to reshape
        a 4D tensor to 2D by calling .view(). However this (dynamic) reshaping
        doesn't work well with ONNX and Int8 tools, and cause using extra
        ops (eg. ExpandDims) that might not be available on mobile.
    Luckily Caffe2 supports 4D tensor for FC, so we can remove those reshape
        after exporting ONNX model.
    """
    from caffe2.python import core

    # find all reshape sub-graph that can be removed, which is now all Reshape
    # sub-graph whose output is only consumed by FC.
    # TODO: to make it safer, we may need the actually value to better determine
    # if a Reshape before FC is removable.
    reshape_sub_graphs = identify_reshape_sub_graph(predict_net)
    sub_graphs_to_remove = []
    for reshape_sub_graph in reshape_sub_graphs:
        reshape_op_id = reshape_sub_graph[-1]
        assert predict_net.op[reshape_op_id].type == "Reshape"
        ssa, _ = core.get_ssa(predict_net)
        reshape_output = ssa[reshape_op_id][1][0]
        consumers = [i for i in range(len(ssa)) if reshape_output in ssa[i][0]]
        if all(predict_net.op[consumer].type == "FC" for consumer in consumers):
            # safety check if the sub-graph is isolated, for this reshape sub-graph,
            # it means it has one non-param external input and one external output.
            ext_inputs, ext_outputs = get_sub_graph_external_input_output(
                predict_net, reshape_sub_graph
            )
            non_params_ext_inputs = [inp for inp in ext_inputs if inp[1] != 0]
            if len(non_params_ext_inputs) == 1 and len(ext_outputs) == 1:
                sub_graphs_to_remove.append(reshape_sub_graph)

    # perform removing subgraph by:
    # 1: rename the Reshape's output to its input, then the graph can be
    #   seen as in-place itentify, meaning whose external input/output are the same.
    # 2: simply remove those ops.
    remove_op_ids = []
    params_to_remove = []
    for sub_graph in sub_graphs_to_remove:
        logger.info(
            "Remove Reshape sub-graph:\n{}".format(
                "".join(
                    ["(#{:>4})\n{}".format(i, predict_net.op[i]) for i in sub_graph]
                )
            )
        )
        reshape_op_id = sub_graph[-1]
        new_reshap_output = predict_net.op[reshape_op_id].input[0]
        rename_op_output(predict_net, reshape_op_id, 0, new_reshap_output)
        ext_inputs, ext_outputs = get_sub_graph_external_input_output(
            predict_net, sub_graph
        )
        non_params_ext_inputs = [inp for inp in ext_inputs if inp[1] != 0]
        params_ext_inputs = [inp for inp in ext_inputs if inp[1] == 0]
        assert len(non_params_ext_inputs) == 1 and len(ext_outputs) == 1
        assert ext_outputs[0][0] == non_params_ext_inputs[0][0]
        assert ext_outputs[0][1] == non_params_ext_inputs[0][1] + 1
        remove_op_ids.extend(sub_graph)
        params_to_remove.extend(params_ext_inputs)

    predict_net = copy.deepcopy(predict_net)
    new_ops = [op for i, op in enumerate(predict_net.op) if i not in remove_op_ids]
    del predict_net.op[:]
    predict_net.op.extend(new_ops)
    for versioned_params in params_to_remove:
        name = versioned_params[0]
        logger.info(
            "Remove params: {} from init_net and predict_net.external_input".format(
                name
            )
        )
        del params[name]
        predict_net.external_input.remove(name)

    return predict_net, params


def fuse_copy_between_cpu_and_gpu(predict_net: caffe2_pb2.NetDef):
    """
    In-place fuse extra copy ops between cpu/gpu for the following case:
        a -CopyAToB-> b -CopyBToA> c1 -NextOp1-> d1
                        -CopyBToA> c2 -NextOp2-> d2
    The fused network will look like:
        a -NextOp1-> d1
          -NextOp2-> d2
    """

    _COPY_OPS = ["CopyCPUToGPU", "CopyGPUToCPU"]

    def _fuse_once(predict_net):
        ssa, blob_versions = core.get_ssa(predict_net)
        consumer_map = get_consumer_map(ssa)
        versioned_external_output = [
            (name, blob_versions[name]) for name in predict_net.external_output
        ]

        for op_id, op in enumerate(predict_net.op):
            if op.type in _COPY_OPS:
                fw_copy_versioned_output = ssa[op_id][1][0]
                consumer_ids = [x[0] for x in consumer_map[fw_copy_versioned_output]]
                reverse_op_type = _COPY_OPS[1 - _COPY_OPS.index(op.type)]

                is_fusable = (
                    len(consumer_ids) > 0
                    and fw_copy_versioned_output not in versioned_external_output
                    and all(
                        predict_net.op[_op_id].type == reverse_op_type
                        and ssa[_op_id][1][0] not in versioned_external_output
                        for _op_id in consumer_ids
                    )
                )

                if is_fusable:
                    for rv_copy_op_id in consumer_ids:
                        # making each NextOp uses "a" directly and removing Copy ops
                        rs_copy_versioned_output = ssa[rv_copy_op_id][1][0]
                        next_op_id, inp_id = consumer_map[rs_copy_versioned_output][0]
                        predict_net.op[next_op_id].input[inp_id] = op.input[0]
                    # remove CopyOps
                    new_ops = [
                        op
                        for i, op in enumerate(predict_net.op)
                        if i != op_id and i not in consumer_ids
                    ]
                    del predict_net.op[:]
                    predict_net.op.extend(new_ops)
                    return True

        return False

    # _fuse_once returns False is nothing can be fused
    while _fuse_once(predict_net):
        pass


def remove_dead_end_ops(net_def: caffe2_pb2.NetDef):
    """remove ops if its output is not used or not in external_output"""
    ssa, versions = core.get_ssa(net_def)
    versioned_external_output = [
        (name, versions[name]) for name in net_def.external_output
    ]
    consumer_map = get_consumer_map(ssa)
    removed_op_ids = set()

    def _is_dead_end(versioned_blob):
        return not (
            versioned_blob in versioned_external_output
            or (
                len(consumer_map[versioned_blob]) > 0
                and all(
                    x[0] not in removed_op_ids for x in consumer_map[versioned_blob]
                )
            )
        )

    for i, ssa_i in reversed(list(enumerate(ssa))):
        versioned_outputs = ssa_i[1]
        if all(_is_dead_end(outp) for outp in versioned_outputs):
            removed_op_ids.add(i)

    # simply removing those deadend ops should have no effect to external_output
    new_ops = [op for i, op in enumerate(net_def.op) if i not in removed_op_ids]
    del net_def.op[:]
    net_def.op.extend(new_ops)
