#!/usr/bin/env python3

import sys

import fblearner_launch_utils as flu
import launch_helper as lh


sys.path.append("mobile-vision/common/tools/")


flu.set_debug_local()


TARGET = "//mobile-vision/projects/model_zoo/tools:collect_models_det"


MODELS = {
    # "search_det_e2e_flops_max_8g/top_ees": ["f155123733"],
    # "search_det_e2e_flops_max_8g/top_seen": ["f155010500"],
    # "search_det_e2e_flops_max_8g_rand_parents_5g_5.3g/top_ees": ["f155493071"],
    # "search_det_e2e_fbnet_b_flops_4_8g/top_seen": ["f157163149"],
    # "search_det_e2e_fbnet_b_flops_4_8g/top_ees": ["f157210911"],
    # "search_det_e2e_fbnet_cse_flops_5_5g/top_seen": ["f157217923"],
    "search_det_e2e_fbnet_cse_flops_5_5g/top_ees": ["f157163138"],
    # "search_det_e2e_fbnet_b_flops_4_8g_person/top_seen": ["f157309963"],
    # "search_det_e2e_fbnet_b_flops_4_8g_person/top_ees": ["f157361928"],
    # "search_det_e2e_fbnet_cse_flops_5_5g_person/top_seen": ["f157256328", "f157352584", "f157354061"],
    # "search_det_e2e_fbnet_cse_flops_5_5g_person/top_ees": ["f157441491"]
}


flu.buck_run("collect_models_det", TARGET, args=["--jobs", lh.json_str(MODELS)])

# ifbpy mobile-vision/projects/model_zoo/scripts/run_collect_models_det.py
