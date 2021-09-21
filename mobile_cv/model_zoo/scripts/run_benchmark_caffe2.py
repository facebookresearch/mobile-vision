#!/usr/bin/env python3

import sys

sys.path.append("mobile-vision/common/tools/")

import aibench_utils as au
import fblearner_launch_utils as flu


flu.set_run_locally()
# flu.set_debug_mode()
# flu.use_prebuilt_binaries()


def get_model_info_caffe2(name, dir_name, res):
    MODEL_PATH = f"/mnt/vol/gfsai-oregon/aml/mobile-vision/model_zoo/models/c2/FBNet/models/{dir_name}/int8"
    model_info = {
        "name": f"c2_{name}_{res}",
        "framework": "caffe2",
        "model_path": MODEL_PATH,
        "model_file": "model.pb",
        "model_init_file": "model_init.pb",
        "inputs": {"data": [1, 3, res, res]},
    }
    return model_info


def run_bench_mobile_caffe2(model_info):
    # benchmark on android
    # see https://our.internmc.facebook.com/intern/wiki/Building-android-on-devservers/
    # for how to setup build for android on devserver
    device = "GalaxyS8US"
    queue = "aibench_interactive"
    au.run_bench(model_info, au.BenchOpts.android(device, queue))


def main():
    TEST_MODELS = [
        ("fbnet_a", "FBNet-A"),
        ("fbnet_b", "FBNet-B"),
        ("fbnet_c", "FBNet-C"),
    ]
    for name, dir_name in TEST_MODELS:
        run_bench_mobile_caffe2(get_model_info_caffe2(name, dir_name, 224))


if __name__ == "__main__":
    main()
