#!/usr/bin/env python3

import sys

sys.path.append("mobile-vision/common/tools/")
import fblearner_launch_utils as flu
import launch_helper as lh


flu.set_debug_local()


TARGET = "//mobile-vision/projects/model_zoo/tools:collect_models"


MODELS = {
    # 07/26/2019
    # "fbnet_a": ["f123927413", "cls"],
    # "fbnet_b": ["f123927409", "cls"],
    # "fbnet_c": ["f123927393", "cls"],
    # "fbnet_ase": ["f124135658", "cls"],
    # "fbnet_bse": ["f124136379", "cls"],
    # "fbnet_cse": ["f124135958", "cls"],
    # # 09/03/2019
    # "fbnet_a": ["f134779565", "cls"],
    # "fbnet_b": ["f134779570", "cls"],
    # "fbnet_c": ["f134779576", "cls"],
    # "fbnet_ase": ["f124499610", "cls"],
    # "fbnet_bse": ["f124699275", "cls"],
    # "fbnet_cse": ["f124499617", "cls"],
    # # int8 friendly models
    # "fbnet_a_i8f": ["f134616454", "cls"],
    # "fbnet_b_i8f": ["f134616456", "cls"],
    # "fbnet_c_i8f": ["f134655804", "cls"],
    # "fbnet_ase_i8f": ["f134616459", "cls"],
    # "fbnet_bse_i8f": ["f134616461", "cls"],
    # "fbnet_cse_i8f": ["f134644352", "cls"],
    # # 12/16/2019, fbnet_v2
    # "fbnet_a": ["f155351132", "cls"],
    # "fbnet_b": ["f153237418", "cls"],
    # "fbnet_c": ["f152910217", "cls"],
    # "fbnet_ase": ["f152910216", "cls"],
    # "fbnet_bse": ["f156832507", "cls"],
    # "fbnet_cse": ["f156832505", "cls"],
    # # int8 friendly models
    # "fbnet_a_i8f": ["f155351131", "cls"],
    # "fbnet_b_i8f": ["f152919289", "cls"],
    # "fbnet_c_i8f": ["f152918373", "cls"],
    # "fbnet_ase_i8f": ["f156799425", "cls"],
    # "fbnet_bse_i8f": ["f157379790", "cls"],
    # "fbnet_cse_i8f": ["f156799593", "cls"],
    # # fbnet v2, int8 model
    # "fbnet_a_i8f_int8_jit": ["f155351131", "cls_jit_int8"],
    # "fbnet_b_i8f_int8_jit": ["f152919289", "cls_jit_int8"],
    # "fbnet_c_i8f_int8_jit": ["f152918373", "cls_jit_int8"],
    # "default": ["f152287703", "cls"],
    # "mnv3": ["f153237412", "cls"],
    # "mnv3_i8f": ["f152919291", "cls"],
    # "dmasking_f1": ["f156819307", "cls"],
    # "dmasking_f4": ["f156704077", "cls"],
    # "dmasking_l2_hs": ["f157689684", "cls"],
    # "dmasking_l3": ["f156819299", "cls"],
    # "dmasking_f2": ["f150930381", "cls"],
    # "dmasking_f3": ["f150722134", "cls"],
    # "dmasking_f4": ["f150512157", "cls"],
    # "FBNetV2_L1": ["f161288344", "cls"],
    # "FBNetV2_L2": ["f161063729", "cls"],
    # "FBNetV2_L3": ["f161089912", "cls"],
    # "FBNetV2_13732M": ["f167473216", "cls"]
    # "eff_2": ["f161287937", "cls"],
    # "eff_3": ["f160154214", "cls"],
    # "eff_4": ["f161101543", "cls"],
    # "FBNetV2_F1": ["f167001958", "cls"],
    # "FBNetV2_F2": ["f167001918", "cls"],
    # "FBNetV2_F3": ["f167001975", "cls"],
    # "FBNetV2_F4": ["f167001913", "cls"],
    # "FBNetV2_L1": ["f183461678", "cls"],
    # "FBNetV2_L2": ["f167001961", "cls"],
    # "FBNetV3_A": ["f235707940", "cls"],
    # "FBNetV3_B": ["f233607847", "cls"],
    # "FBNetV3_C": ["f234529537", "cls"],
    # "FBNetV3_D": ["f231225742", "cls"],
    # "FBNetV3_E": ["f230158956", "cls"],
    # "FBNetV3_F": ["f230161092", "cls"],
    "FBNetV3_F": ["f244000132", "cls"],
    "FBNetV3_G": ["f241670737", "cls"]
    # "FBNetV3_G": ["f226438997", "cls"],
}


flu.buck_run("collect_models", TARGET, args=["--jobs", lh.json_str(MODELS)])
