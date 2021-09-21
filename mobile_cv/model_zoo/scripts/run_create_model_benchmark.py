#!/usr/bin/env python3

import datetime
import os
import sys

sys.path.append("mobile-vision/common/tools/")
import tempfile

import aibench_utils as au
import fblearner_launch_utils as flu
import launch_helper as lh


flu.set_run_locally()
# flu.set_debug_mode()
# flu.use_prebuilt_binaries()


CREATE_MODEL_TARGET = "mobile-vision/mobile_cv/mobile_cv/model_zoo/tools:create_model"


BENCHMARK_DEVICE_SERVER = "Skylake_I_tw"
BENCHMARK_DEVICE_ANDROID = "GalaxyS8US"
BENCHMARK_TYPES = [
    # ("server_local", BENCHMARK_DEVICE_SERVER),           # benchmark locally
    # ("server_aibench", BENCHMARK_DEVICE_SERVER),         # python based, has per op latency
    # ("server_aibench", "GPU"),         # python based, has per op latency
    # ("server_aibench_cpp", BENCHMARK_DEVICE_SERVER),       # total runtime only, but accurate
    ("android_aibench", BENCHMARK_DEVICE_ANDROID),  # android
    (
        "android_lite_aibench",
        BENCHMARK_DEVICE_ANDROID,
    ),  # android_lite, per layer latency
]


def create_model(name, base_wd, builder, arch, inputs, **kwargs):
    wd = os.path.join(base_wd, name)
    task_args = [
        "--builder",
        builder,
        "--data_shape",
        lh.json_str(inputs),
        "--output_dir",
        wd,
        "--int8_backend",
        "fbgemm",
        # "qnnpack",
        "--self_contained",
        1,
        #  "--fuse_bn", 0,
    ]
    if arch is not None:
        task_args += [
            "--arch_name",
            arch,
        ]
    if kwargs:
        task_args += ["--arch_args", lh.json_str(kwargs)]

    flu.buck_run(name, CREATE_MODEL_TARGET, args=task_args)
    return wd


def run(model_info, name_prefix=""):
    name = model_info["name"]
    model_type = model_info.pop("model_type")
    base_wd = tempfile.mkdtemp()
    print(f"Creating {name}...")
    model_path = create_model(base_wd=base_wd, **model_info)
    # model_path = "/tmp/tmph7kn2n6k/fbnet_c/"
    print(f"Benchmarking {name} int8...")
    for bench_type, device in BENCHMARK_TYPES:
        if bench_type in ("android_aibench", "android_lite_aibench"):
            au.BenchJit(bench_type, device).run_with_shape(
                name_prefix + name,
                os.path.join(model_path, model_type, "model.jit"),
                input_shapes=model_info["inputs"],
                optimize_for_mobile=False,
            )
        else:
            au.BenchJit(bench_type, device, on_gpu="GPU" in device).run_with_shape(
                name_prefix + name,
                os.path.join(model_path, model_type, "model.jit"),
                input_shapes=model_info["inputs"],
                optimize_for_mobile=False,
            )


MODELS = [
    # ("fbnet_v2", "fbnet_a", 224, "int8_jit"),
    # ("fbnet_v2", "fbnet_b", 224, "int8_jit"),
    # ("fbnet_v2", "fbnet_c", 224, "int8_jit"),
    # ("fbnet_v2", "fbnet_ase", 224, "int8_jit"),
    # ("fbnet_v2", "fbnet_bse", 224, "int8_jit"),
    # ("fbnet_v2", "fbnet_cse", 224, "int8_jit"),
    # ("fbnet_v2", "default", 224, "int8_jit"),
    # ("fbnet_v2", "fbnet_a", 224, "jit"),
    # ("fbnet_v2", "fbnet_b", 224, "jit"),
    # ("fbnet_v2", "fbnet_c", 224, "jit"),
    # ("fbnet_v2", "fbnet_ase", 224, "jit"),
    # ("fbnet_v2", "fbnet_bse", 224, "jit"),
    # ("fbnet_v2", "fbnet_cse", 224, "jit"),
    # ("fbnet_v2", "default", 224, "jit"),
    # ("fbnet_v2", "dmasking_f1", 128, "int8_jit"),
    # ("fbnet_v2", "dmasking_f2", 160, "int8_jit"),
    # ("fbnet_v2", "dmasking_f3", 192, "int8_jit"),
    # ("fbnet_v2", "dmasking_f4", 224, "int8_jit"),
    # ("fbnet_v2", "dmasking_f5", 256, "int8_jit"),
    # ("fbnet_v2", "dmasking_l3", 288, "int8_jit"),
    # ("fbnet_v2", "mnv3", 224, "int8_jit"),
    # ("fbnet_v2", "eff_0", 224, "int8_jit"),
    # ("fbnet_v2", "eff_1", 240, "int8_jit"),
    # ("fbnet_v2", "eff_2", 260, "int8_jit"),
    # ("fbnet_v2", "eff_3", 300, "int8_jit"),
    # ("fbnet_v2", "FBNetV2_462M", 256, "int8_jit"),
    # ("fbnet_v2", "FBNetV2_790M", 256, "int8_jit"),
    # ("fbnet_v2", "FBNetV2_908M", 268, "int8_jit"),
    # ("fbnet_v2", "FBNetV2_1500M", 288, "int8_jit"),
    # ("fbnet_v2", "FBNetV2_4672M", 380, "int8_jit"),
    ("fbnet_v2_backbone", "LeViT_192_noDistill", 224, "jit"),
    ("fbnet_v2_backbone", "LeViT_192_noDistill", 224, "int8_jit"),
    ("fbnet_v2", "FBNetV3_D", 248, "jit"),
    ("fbnet_v2", "FBNetV3_D", 248, "int8_jit"),
    # ("classy_resnext101_32x4d", None, 224, "jit"),
    # int8 not supported
    # ("fbnet_v1", "FBNetV2_L1", 260),
    # ("fbnet_v1", "FBNetV2_L2", 300),
    # ("fbnet_v1", "FBNetV2_L3", 380),
    # int8 not supported
    # ("resnet50", None, 224, "jit"),
    # ("resnet101", None, 224, "jit"),
    # ("resnet152", None, 224, "jit"),
    # ("resnext50_32x4d", None, 224, "jit"),
    # ("resnext101_32x8d", None, 224, "jit"),
]
MODEL_INFOS = [
    {
        "name": f"{builder}{'_' + x if x is not None else ''}",
        "builder": builder,
        "arch": x,
        "inputs": [[1, 3, res, res]],
        "model_type": model_type,
    }
    for builder, x, res, model_type in MODELS
]


prefix_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
for x in MODEL_INFOS:
    run(x, prefix_str + "_")
