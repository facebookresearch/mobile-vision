#!/usr/bin/env python3

import sys

import fblearner_launch_utils as flu


sys.path.append("mobile-vision/common/tools/")



# flu.set_debug_local()


TARGET = "//mobile-vision/projects/open-source/mobile-vision/ChamNet/code/fb:eval_model"
DATASET = "/data/local/packages/ai-group.imagenet-full-size/prod/imagenet_full_size/"


def run(name, model_info, parent=None):
    print("Evaluting {}".format(name))
    args = ["--data", DATASET, "--max_images", 0]
    for mname, mval in model_info.items():
        args += [f"--{mname}", mval]

    gpu = 1
    if model_info.get("dist_backend", "gloo") == "nccl":
        gpu = model_info.get("num_processes", 8)
    flu.buck_run(
        "eval_" + name,
        TARGET,
        args,
        parent=parent,
        # deps=[model_dir],
        resources=flu.Resources(gpu=gpu),
    )


def run_models(base_name, models, parent=None):
    for x in models:
        run(base_name + "_" + x, models[x], parent=parent)


# fbnet v1 builder
MODEL_NAMES_FBNET_V1_DMASKING = [
    ("dmasking_f2", 160, 183),
    ("dmasking_f3", 192, 219),
    ("dmasking_f4", 224, 256),
]
MODELS_MODEL_ZOO_FBNET_V1_DMASKING = {
    x: {
        "model_type": "model_zoo",
        "model_path": f"fbnet_v1/{x}",
        "crop_res": res,
        "resize_res": resize_res,
    }
    for x, res, resize_res in MODEL_NAMES_FBNET_V1_DMASKING
}

MODEL_NAMES_FBNET_V1_FBNET_LARGE = [
    ("FBNetV2_L1", 260, 280),
    ("FBNetV2_L2", 300, 316),
    ("FBNetV2_L3", 380, 397),
    ("eff_3", 300, 332),
]
MODELS_MODEL_ZOO_FBNET_V1_FBNET_LARGE = {
    x: {
        "model_type": "model_zoo",
        "model_path": f"fbnet_v1/{x}",
        "crop_res": res,
        "resize_res": resize_res,
        "num_processes": 8,
        "batch_size": 32,
        "dist_backend": "nccl",
    }
    for x, res, resize_res in MODEL_NAMES_FBNET_V1_FBNET_LARGE
}


# fbnet v2 builder
MODEL_NAMES = [
    "fbnet_a",
    "fbnet_b",
    "fbnet_c",
    "fbnet_ase",
    "fbnet_bse",
    "fbnet_cse",
    "fbnet_a_i8f",
    "fbnet_b_i8f",
    "fbnet_c_i8f",
    "fbnet_ase_i8f",
    "fbnet_bse_i8f",
    "fbnet_cse_i8f",
    "default",
    "mnv3",
    "mnv3_i8f",
]
MODELS_MODEL_ZOO = {
    x: {"model_type": "model_zoo", "model_path": f"fbnet_v2/{x}", "crop_res": 224}
    for x in MODEL_NAMES
}


MODEL_NAMES_DMASKING = [
    ("dmasking_f1", 128, 146),
    ("dmasking_f4", 224, 256),
    ("dmasking_l2_hs", 256, 288),
    ("dmasking_l3", 288, 324),
]
MODELS_MODEL_ZOO_DMASKING = {
    x: {
        "model_type": "model_zoo",
        "model_path": f"fbnet_v2/{x}",
        "crop_res": res,
        "resize_res": resize_res,
    }
    for x, res, resize_res in MODEL_NAMES_DMASKING
}


# jit model
MODEL_NAMES_JIT = [
    "fbnet_a_i8f_int8_jit",
    "fbnet_b_i8f_int8_jit",
    "fbnet_c_i8f_int8_jit",
]
MODELS_MODEL_ZOO_JIT = {
    x: {
        "model_type": "model_zoo",
        "model_path": f"jit/{x}",
        "crop_res": 224,
        "resize_res": 256,
        # currently dpp is not supported for jit model
        "num_processes": 1,
        "int8_backend": "qnnpack",
        "batch_size": 128,
    }
    for x in MODEL_NAMES_JIT
}


# pytorch model
MODEL_NAMES_TORCHVISION = [
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
]
MODELS_MODEL_ZOO_TORCHVISION = {
    x: {
        "model_type": "model_zoo",
        "model_path": x,
        "crop_res": 224,
        "resize_res": 256,
        "num_processes": 8,
        "batch_size": 32,
        "dist_backend": "nccl",
    }
    for x in MODEL_NAMES_TORCHVISION
}


with flu.Workflow(
    "eval_fbnet_model_zoo", ctype="parallel", secure_group_name="team_ai_mobile_cv"
) as wf:
    wf.set_entitlement("default_vll_gpu")
    run_models("fbnet_v1", MODELS_MODEL_ZOO_FBNET_V1_DMASKING, wf)
    run_models("fbnet_v1", MODELS_MODEL_ZOO_FBNET_V1_FBNET_LARGE, wf)
    run_models("fbnet_v2", MODELS_MODEL_ZOO, wf)
    run_models("fbnet_v2", MODELS_MODEL_ZOO_DMASKING, wf)
    run_models("jit", MODELS_MODEL_ZOO_JIT, wf)
    run_models("torchvision", MODELS_MODEL_ZOO_TORCHVISION, wf)
