#!/usr/bin/env python3

import copy
import functools
import os

from caffe2.python import net_drawer


def _modify_blob_names(ops, blob_rename_f):
    ret = []

    def _replace_list(blob_list, replaced_list):
        del blob_list[:]
        blob_list.extend(replaced_list)

    for x in ops:
        cur = copy.deepcopy(x)
        _replace_list(cur.input, list(map(blob_rename_f, cur.input)))
        _replace_list(cur.output, list(map(blob_rename_f, cur.output)))
        ret.append(cur)

    return ret


def _rename_blob(name, blob_sizes, blob_ranges):
    def _list_to_str(bsize):
        ret = ", ".join([str(x) for x in bsize])
        ret = "[" + ret + "]"
        return ret

    ret = name
    if blob_sizes is not None and name in blob_sizes:
        ret += "\n" + _list_to_str(blob_sizes[name])
    if blob_ranges is not None and name in blob_ranges:
        ret += "\n" + _list_to_str(blob_ranges[name])

    return ret


# graph_name could not contain word 'graph'
def save_graph(
    net, file_name, graph_name="net", op_only=True, blob_sizes=None, blob_ranges=None
):
    blob_rename_f = functools.partial(
        _rename_blob, blob_sizes=blob_sizes, blob_ranges=blob_ranges
    )
    return save_graph_base(net, file_name, graph_name, op_only, blob_rename_f)


def save_graph_base(
    net, file_name, graph_name="net", op_only=True, blob_rename_func=None
):
    graph = None
    ops = net.op
    if blob_rename_func is not None:
        ops = _modify_blob_names(ops, blob_rename_func)
    if not op_only:
        graph = net_drawer.GetPydotGraph(ops, graph_name, rankdir="TB")
    else:
        graph = net_drawer.GetPydotGraphMinimal(
            ops, graph_name, rankdir="TB", minimal_dependency=True
        )

    try:
        par_dir = os.path.dirname(file_name)
        if not os.path.exists(par_dir):
            os.makedirs(par_dir)

        format = os.path.splitext(os.path.basename(file_name))[-1]
        if format == ".png":
            graph.write_png(file_name)
        elif format == ".pdf":
            graph.write_pdf(file_name)
        elif format == ".svg":
            graph.write_svg(file_name)
        else:
            print("Incorrect format {}".format(format))
    except Exception as e:
        print("Error when writing graph to image {}".format(e))

    return graph
