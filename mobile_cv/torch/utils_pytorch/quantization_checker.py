#!/usr/bin/env python3

"""
Tools help to check/visualze the histogram of the model output
"""

import copy
import logging
from typing import Any, Dict, List, NamedTuple, Optional

import matplotlib.pyplot as plt
import mobile_cv.torch.utils_pytorch.vis as vis
import plotly.express as px
import plotly.graph_objects as go
import torch
from tensorboard.fb.plotly import add_plotly
from torch.ao.quantization import (
    default_weight_observer,
    ObserverBase,
    prepare,
    QConfig,
)
from torch.ao.quantization.quantize_fx import prepare_fx


logger = logging.getLogger(__name__)


class RecordingObserver(ObserverBase):
    def __init__(self, **kwargs):
        super(RecordingObserver, self).__init__(dtype=None, **kwargs)
        self.tensor_val = []

    def forward(self, x):
        self.tensor_val.append(copy.deepcopy(x))
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for RecordingObserver")

    @torch.jit.export
    def get_tensor_value(self):
        return self.tensor_val


def _add_observers(model):
    def _hook(module, inputs, outputs):
        assert hasattr(module, "_activation_observer")
        module._activation_observer(outputs)

    def _func(module):
        assert not hasattr(module, "_activation_observer")
        module._activation_observer = RecordingObserver()
        module._activation_observer_hook_handle = module.register_forward_hook(_hook)

    model.apply(_func)
    return model


def add_stats_observers(model, sub_modules: Optional[List[str]] = None):
    if sub_modules is not None:
        for smn in sub_modules:
            sm = model.get_submodule(smn)
            sm = _add_observers(sm)
            assign_module(model, smn, sm)
    else:
        model = _add_observers(model)
    return model


def get_observers(model):
    ret = {}
    for name, module in model.named_modules():
        if isinstance(module, RecordingObserver):
            ret[name] = module
    return ret


def get_observer_config():
    act_observer = RecordingObserver
    return QConfig(weight=default_weight_observer, activation=act_observer)


def add_stats_observers_quantization_api(model):
    """Using quantization api to add stats obserers to models"""
    model.qconfig = get_observer_config()
    model = prepare(model)
    return model


def assign_module(model, submodule_name, submodule):
    sub_names = submodule_name.split(".")
    if len(sub_names) == 1:
        parent = model
    else:
        parent = model.get_submodule(".".join(sub_names[:-1]))

    setattr(parent, sub_names[-1], submodule)


def add_stats_observers_fx(model, sub_modules: Optional[List[str]] = None):
    """Using quantization api to add stats obserers to models"""
    qconfig_dict = {"": get_observer_config()}
    if sub_modules is not None:
        for smn in sub_modules:
            sm = model.get_submodule(smn)
            sm = prepare_fx(sm, qconfig_dict)
            assign_module(model, smn, sm)
    else:
        model = prepare_fx(model)
    return model


def flatten_dict(adict):
    """Convert a nested dict to non-nested one.
    {"ab": 123, "cd": {"ef": 1, "gh": 2}} -> {"ab": 123, "cd.ef": 1, "cd.gh": 2}
    """
    import pandas as pd

    df = pd.json_normalize(adict, sep=".")
    return df.to_dict(orient="records")[0]


def _is_list_of_dict(obj):
    return isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict)


def list_of_dict_to_dict_of_list(list_of_dict):
    assert _is_list_of_dict(list_of_dict), type(list_of_dict)
    return {k: [dic[k] for dic in list_of_dict] for k in list_of_dict[0]}


def convert_list_of_dict(adict):
    """Convert list of dicts to dict of lists"""
    import mobile_cv.common.misc.iter_utils as iu

    def _is_list(obj):
        if _is_list_of_dict(obj):
            return False
        return isinstance(obj, list)

    riter = iu.recursive_iterate(
        adict,
        seq_check_func=_is_list,
        wait_on_send=True,
    )
    for item in riter:
        if _is_list_of_dict(item):
            item = list_of_dict_to_dict_of_list(item)
        riter.send(item)

    return riter.value


def _to_float(value):
    if value.dtype == torch.quint8:
        dq = torch.nn.quantized.DeQuantize()
        return dq(value)
    return value


def get_stats_results(model):
    observer_dict = get_observers(model)

    observer_dict = {x: y.tensor_val for x, y in observer_dict.items()}

    observer_dict = convert_list_of_dict(observer_dict)
    observer_dict = flatten_dict(observer_dict)
    observer_dict = {
        x: (
            y
            if (isinstance(y, list) and all(isinstance(v, torch.Tensor) for v in y))
            else None
        )
        for x, y in observer_dict.items()
    }

    ret = {
        x: torch.cat([_to_float(v).reshape(-1) for v in y], dim=0) if y else None
        for x, y in observer_dict.items()
    }
    return ret


def visualize_stats_results(model, results, groups=None, log_dir=None):
    tb = vis.get_tensorboard("visualze_stats", log_dir=log_dir)
    add_visualize_stats_results(tb, step=0, model=model, results=results, groups=groups)
    print(vis.get_on_demand_url(tb.log_dir))


def _get_histogram_figure(values):
    fig = plt.figure()
    # plt.hist(values, bins="auto")
    plt.hist(values, bins=2048)
    return fig


def _get_histogram_plotly(values):
    if values is None:
        return None

    fig = px.histogram(values, nbins=1024)
    return fig


def _get_histogram_plotly_compare(values_dict: Dict[str, torch.Tensor]):
    any_result = False
    fig = go.Figure()
    for name, values in values_dict.items():
        print(type(values))
        if values is not None:
            fig.add_trace(go.Histogram(name=name, x=values))
            any_result = True

    # Overlay both histograms
    fig.update_layout(barmode="overlay")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    return fig if any_result else None


def _run_func(item):
    print(f"Getting {item[0]}")
    return item[0], _get_histogram_plotly(item[1])


def _get_histogram_plotly_parallel(adict):
    from multiprocessing import Pool

    with Pool(processes=32) as pool:
        out = pool.map(_run_func, adict.items())

    return {x[0]: x[1] for x in out}


def _run_func_compare(item):
    print(f"Getting {item[0]}")
    return item[0], _get_histogram_plotly_compare(item[1])


def _get_histogram_plotly_compare_parallel(results: Dict[str, Dict[str, Any]]):
    from multiprocessing import Pool

    results_by_sub_items = {}

    for name, cur_result in results.items():
        for sub_item_name, sub_item in cur_result.items():
            if sub_item_name not in results_by_sub_items:
                results_by_sub_items[sub_item_name] = {}
            results_by_sub_items[sub_item_name].update({name: sub_item})

    with Pool(processes=32) as pool:
        out = pool.map(_run_func_compare, results_by_sub_items.items())

    return {x[0]: x[1] for x in out}


def add_visualize_stats_results(
    logger,
    step,
    model,
    results,
    groups=None,
    use_plotly=False,
):
    logger.add_text(f"model_{step}", str(model))

    def _get_with_group_name(name, groups):
        if groups is None:
            return name
        for gn in groups:
            if name.startswith(gn):
                return f"{gn}/{name}"
        return name

    if use_plotly:
        results_plot = _get_histogram_plotly_parallel(results)

    for name, values in results.items():
        gname = _get_with_group_name(name, groups)
        print(f"Adding histogram for {name}...")
        if values is not None:
            if not use_plotly:
                logger.add_histogram(gname, values, global_step=step)
            else:
                # for debugging
                # logger.add_figure(gname + "_fig", _get_histogram_figure(values), global_step=step)
                # add_plotly(logger, gname + f"_step_{step}", _get_histogram_plotly(values))
                add_plotly(logger, gname + f"_step_{step}", results_plot[name])

        else:
            print(f"Skipping {name} as it is None")


class Results(NamedTuple):
    model: torch.nn.Module
    results: Dict[str, Any]
    group_names: List[str]


def add_visualize_stats_results_plotly(
    logger,
    results: Dict[str, Results],
):
    for name, values in results.items():
        logger.add_text(f"model_{name}", str(values.model))

    def _get_with_group_name(name, groups):
        if groups is None:
            return name
        for gn in groups:
            if name.startswith(gn):
                return f"{gn}/{name}"
        return name

    results_for_plots = {x: y.results for x, y in results.items()}
    results_plot = _get_histogram_plotly_compare_parallel(results_for_plots)

    groups = next(iter(results.values())).group_names

    for name, values in results_plot.items():
        gname = _get_with_group_name(name, groups)
        print(f"Adding histogram for {name}...")
        if values is not None:
            add_plotly(logger, gname + "_pl", results_plot[name])
        else:
            print(f"Skipping {name} as it is None")
