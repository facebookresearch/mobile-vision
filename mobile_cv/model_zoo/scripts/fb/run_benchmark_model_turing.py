#!/usr/bin/env python3

import json
import sys

import fblearner_launch_utils as flu


sys.path.append("mobile-vision/common/tools/")


flu.set_debug_local()


TARGET = "//mobile-vision/projects/model_zoo/tools:benchmark_model_turing"


MODELS = [
    ["FBNetV3_A_paper", (1, 3, 224, 224)],
    ["FBNetV3_G_paper", (1, 3, 320, 320)],
    ["FBNetV2_F4", (1, 3, 224, 224)],
]

for model in MODELS:
    name, input_shape = model
    # arg = json.dumps(' '.join(["--arch_name", name, "--data_shape", input_shape]))
    flu.buck_run(
        "benchmark_model_turing",
        TARGET,
        args=[
            "--arch_name",
            name,
            "--data_shape",
            json.dumps(input_shape, separators=(",", ":")),
        ],
    )

# ifbpy mobile-vision/projects/model_zoo/scripts/run_benchmark_model_turing.py
