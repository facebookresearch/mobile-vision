#!/usr/bin/env python3
import argparse
import gc
import os
import time

import numpy as np
import torch
from numpy import percentile as np_pctile


"""
Example run:
buck run @mode/opt mobile-vision/projects/model_zoo/tools:jit_speed_benchmark -- --model \
/mnt/vol/gfsai-oregon/aml/mobile-vision/model_zoo/models/20200103/fACAVX/fbnet_c_i8f_int8_jit_f152918373/model.jpt \
--input_dims 1,3,224,224 --torch_threads 1 --int8_backend fbgemm
"""


def parse_args():
    parser = argparse.ArgumentParser(description="jit speed benchmark")
    parser.add_argument("--model", type=str)
    parser.add_argument("--input_dims", type=str, default=None)
    parser.add_argument("--input_type", type=str, default="float")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--iter", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check_freq", type=int, default=1)
    parser.add_argument("--no_autograd_profiling", action="store_true", default=False)
    parser.add_argument("--report_pep", action="store_true", default=False)
    parser.add_argument("--run_garbage_collector", action="store_true", default=False)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--int8_backend", type=str, default=None)
    parser.add_argument("--flush_denormal", action="store_true", default=False)
    parser.add_argument("--on_gpu", action="store_true", default=False)
    args = parser.parse_args()
    return args


def parse_input_dims(args):
    """Parse input from input_dims
    For input_dims, the format: 1,3,224,224;...
    """
    if args.input_dims is None:
        return None

    assert isinstance(args.input_dims, str)
    input_dims = args.input_dims.split(";")

    def _parse_single(dims):
        parsed = dims.split(",")
        parsed = [int(x) for x in parsed]
        return parsed

    input_dims = [_parse_single(x) for x in input_dims]

    # parse input types
    input_types = args.input_type.split(";")
    assert len(input_dims) == len(
        input_types
    ), f"input_dims {args.input_dims} and input_type {args.input_type} size not match"
    assert all(x == "float" for x in input_types), "Only float type is supported"

    # create input tensors
    ret = []
    for cur_dim, cur_type in zip(input_dims, input_types):
        assert cur_type == "float"
        cur = torch.zeros(cur_dim)
        ret.append(cur)

    return ret


def parse_input_file(args):
    """Parse input from input_file"""
    if args.input_file is None:
        return None

    ret = torch.load(args.input_file, map_location="cpu")
    return ret


def parse_inputs(args):
    """Input could be either from a file or specify by input_dims
    For input_file, it is a torch file
    For input_dims, the format: 1,3,224,224;...
    """
    ret = parse_input_file(args)
    if ret is not None:
        return ret
    ret = parse_input_dims(args)
    if ret is not None:
        return ret

    print("No input is provided.")
    return []


def log_check_info(i, runtimes, timer, prefix=""):
    rt = np.array(runtimes)
    p50, p75, p99 = (np_pctile(rt, 50), np_pctile(rt, 75), np_pctile(rt, 90))
    print(
        "{}: Ran {} trials with {:.2f}ms, avg:{:.2f}ms, p50:{:.2f}ms, p75:{:.2f}ms, p99:{:.2f}ms .".format(
            prefix, i, 1000 * timer.diff, 1000 * timer.average_time, p50, p75, p99
        )
    )


class Timer(object):
    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=False):
        self.diff = time.time() - self.start_time
        self.calls += 1
        self.total_time += self.diff
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def maybe_run_autograd_profile(args, run_model):
    if args.no_autograd_profiling:
        return
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        run_model()
    print(
        "autograd prof:\n {} \n".format(prof.key_averages(group_by_input_shape=False))
    )
    print("autograd prof table:\n {} \n".format(prof.table(row_limit=1000)))


def maybe_ai_pep_output(args, runtimes):
    if not args.report_pep:
        return

    # @dep=//aibench/oss/benchmarking:aibench_observer_fbcode
    from aibench_observer.utils.observer import emitMetric

    for runtime in runtimes:
        print(
            emitMetric(
                "PyTorchObserver",
                type="NET",
                metric="latency",
                unit="ms_per_iter",
                value=str(runtime),
            )
        )


def bench_model(run_model, iter, check_freq, prefix, run_garbage_collector):
    timer = Timer()
    runtimes = []

    for i in range(iter):
        if i and i % check_freq == 0:
            log_check_info(i, runtimes=runtimes, timer=timer, prefix=prefix)

        if run_garbage_collector:
            gc.collect()

        timer.tic()
        run_model()
        runtimes.append(1000 * timer.toc(average=False))

    return runtimes


def init_env(args):
    os.environ["PYTHONUNBUFFERED"] = "1"

    if args.torch_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(args.torch_threads)
        print(f"Use {args.torch_threads} threads.")

    if args.int8_backend is not None:
        torch.backends.quantized.engine = args.int8_backend
        print(f"Use {args.int8_backend} backend.")

    if args.flush_denormal:
        torch.set_flush_denormal(True)
        print("Use flush_denormal = True")


def move_to_device(data, device):
    if isinstance(data, dict):
        return {x: move_to_device(y, device) for x, y in data.items()}
    if isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def main():
    args = parse_args()

    init_env(args)

    # load jit model
    model = torch.jit.load(args.model)
    # prepare input
    input_data = parse_inputs(args)

    def run_model():
        return model(*input_data)

    if args.on_gpu:
        model.cuda()
        input_data = move_to_device(input_data, "cuda")

    # run warmup trials
    print("Warming up...")
    bench_model(
        run_model,
        args.warmup,
        args.check_freq,
        prefix="WARMUP",
        run_garbage_collector=args.run_garbage_collector,
    )
    # run real trials
    print("Benchmarking...")
    runtimes = bench_model(
        run_model,
        args.iter,
        args.check_freq,
        prefix="RUN",
        run_garbage_collector=args.run_garbage_collector,
    )

    # per op profiling
    maybe_run_autograd_profile(args, run_model)
    maybe_ai_pep_output(args, runtimes)


if __name__ == "__main__":
    main()
