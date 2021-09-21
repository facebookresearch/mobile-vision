#!/usr/bin/env python3

import sys

sys.path.append("mobile-vision/common/tools/")
import fblearner_launch_utils as flu
import launch_helper as lh


flu.set_debug_local()


TARGET = "//mobile-vision/projects/model_zoo/tools:collect_models_cls"


MODELS = {
    # "search_cls_e2e_default_flops_300m_360m_lr0.4_node1/top_seen": ["f144849358"],
    # "search_cls_e2e_default_flops_300m_360m_lr0.4_node1/top_ees": ["f144851074"],
    # "search_cls_e2e_default_flops_300m_360m/top_seen": ["f149814442"],
    # "search_cls_e2e_default_flops_300m_360m/top_ees": ["f150532260"],
    # "search_cls_e2e_model_size_3m_5m_cham_f_0.7x_lr0.4_node1/top_ees": ["f146001452"],
    # "search_cls_e2e_model_size_3m_5m_cham_f_0.7x_lr0.4_node1/top_seen": ["f146455308"],
    # "search_cls_e2e_model_size_3m_5m_cham_f_0.7x/top_ees": ["f146708123"],
    # "search_cls_e2e_model_size_3m_5m_cham_f_0.7x/top_seen": ["f146698144"],
    # "search_cls_e2e_model_size_5m_cham_f_0.7x_res128/top_ees": ["f147284898"],
    # "search_cls_e2e_model_size_5m_cham_f_0.7x_res128/top_seen": ["f147280771"],
    # "search_cls_e2e_default_100k/top_ees": ["f149351477", "f149351064", "f149351379", "f148086118"],
    # "search_cls_e2e_default_100k/top_seen": ["f148084815"],
    # "search_cls_e2e_default_100k/top5_ees_rand_parents_180_220M": ["f148216405"],
    # "search_cls_e2e_default_100k/top12_ees_rand_parents_300_360M": ["f148903454", "f148220904"],
    # "search_cls_e2e_default_100k/top5_ees_rand_parents_600_700M": ["f148902914", "f148198590"],
    # "search_cls_e2e_flops_30m_100m_cham_e_res160/top_seen": ["f149347922"],
    # "search_cls_e2e_flops_30m_100m_cham_e_res160/top_ees": ["f149048787"],
    # "search_cls_e2e_flops_1m_30m_cham_f_res128/top_seen": ["f149080705"],
    # "search_cls_e2e_flops_1m_30m_cham_f_res128/top_ees": ["f149080500"],
    # "search_cls_e2e_flops_100m_300m_default_as148216879/top_seen": ["f150217456"],
    # "search_cls_e2e_flops_100m_300m_default_as148216879/top_ees": ["f150156439"],
    # "search_cls_e2e_flops_300m_1g_default_as148199061/top_seen": ["f152471411"],
    # "search_cls_e2e_flops_300m_1g_default_as148199061/top_ees": ["f151506016", "f151507109"],
    # "search_cls_e2e_flops_1g_2g_default_as148085623/top_ees": ["f152463128"],
    "search_cls_e2e_flops_1g_2g_default_as148085623/top_seen": ["f152469507"],
}


flu.buck_run("collect_models_cls", TARGET, args=["--jobs", lh.json_str(MODELS)])

# ifbpy mobile-vision/projects/model_zoo/scripts/run_collect_models_cls.py
