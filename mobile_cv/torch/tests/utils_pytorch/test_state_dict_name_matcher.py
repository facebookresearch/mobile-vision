#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import unittest

import mobile_cv.torch.utils_pytorch.state_dict_name_matcher as sdnm
import torch


class TestUtilsPytorchStateDictNameMatcher(unittest.TestCase):
    def test_matcher(self):
        sd1 = {
            "model.m1.aa.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.ab.bias": torch.zeros(1, 3, 4, 4),
            "model.m2.conv.weight": torch.zeros(1, 3, 4, 4),
            "model.m2.bn.weight": torch.zeros(1, 3, 4, 4),
            "model.m2.relu.weight": torch.zeros(1, 3, 4, 4),
        }
        sd2 = {
            "model.m2_conv.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.aa_post.weight": torch.zeros(1, 3, 4, 4),
            "model.m2_bn.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.ab_post.bias": torch.zeros(1, 3, 4, 4),
            "model.m2_relu.weight": torch.zeros(1, 3, 4, 4),
        }

        ret = sdnm.get_state_dict_name_mapping(sd1, sd2)
        self.assertEqual(
            ret,
            {
                "model.m1.aa.weight": "model.m1.aa_post.weight",
                "model.m1.ab.bias": "model.m1.ab_post.bias",
                "model.m2.conv.weight": "model.m2_conv.weight",
                "model.m2.bn.weight": "model.m2_bn.weight",
                "model.m2.relu.weight": "model.m2_relu.weight",
            },
        )

    def test_matcher_shape(self):
        sd1 = {
            "model.m1.aa.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.bb.weight": torch.zeros(1, 3, 2, 2),
        }
        sd2 = {
            "model.m1.ccb.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.dda.weight": torch.zeros(1, 3, 2, 2),
        }

        ret = sdnm.get_state_dict_name_mapping(sd1, sd2)
        self.assertEqual(
            ret,
            {
                "model.m1.aa.weight": "model.m1.ccb.weight",
                "model.m1.bb.weight": "model.m1.dda.weight",
            },
        )

    def test_list_mapping(self):
        list1 = [
            "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:12:48_0.8357651039198697_06-16-2023_20:12:49.png",
            "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:12:48_0.8357651039198697_hint_06-16-2023_20:12:51.png",
            "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:12:48_0.8357651039198697_mask_06-16-2023_20:12:53.png",
            "gpu0_b_voree_vbtmt_otym_t_trmrtr_gmeritm_wx_n_rmgri_006-16-2023_20:10:05_0.763774618976614_06-16-2023_20:10:07.png",
        ]

        list2 = [
            "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:14:05_0.8357651039198697_06-16-2023_20:14:06.png",
            "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:14:05_0.8357651039198697_hint_06-16-2023_20:14:08.png",
            "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:14:05_0.8357651039198697_mask_06-16-2023_20:14:10.png",
            "gpu0_b_voree_vbtmt_otym_t_trmrtr_gmeritm_wx_n_rmgri_006-16-2023_20:11:22_0.763774618976614_06-16-2023_20:11:24.png",
            "gpu0_b_voree_vbtmt_otym_t_trmrtr_gmeritm_wx_n_rmgri_006-16-2023_20:11:22_0.763774618976614_hint_06-16-2023_20:11:28.png",
            "gpu0_b_voree_vbtmt_otym_t_trmrtr_gmeritm_wx_n_rmgri_006-16-2023_20:11:22_0.763774618976614_mask_06-16-2023_20:11:30.png",
        ]

        mapping = sdnm.get_list_mapping(list1, list2)
        self.assertDictEqual(
            mapping,
            {
                "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:12:48_0.8357651039198697_06-16-2023_20:12:49.png": "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:14:05_0.8357651039198697_06-16-2023_20:14:06.png",
                "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:12:48_0.8357651039198697_hint_06-16-2023_20:12:51.png": "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:14:05_0.8357651039198697_hint_06-16-2023_20:14:08.png",
                "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:12:48_0.8357651039198697_mask_06-16-2023_20:12:53.png": "gpu0_l_ggmte_zp_ae_g_ziemnv_iv_g_irmcw_kged_abcdefghij_006-16-2023_20:14:05_0.8357651039198697_mask_06-16-2023_20:14:10.png",
                "gpu0_b_voree_vbtmt_otym_t_trmrtr_gmeritm_wx_n_rmgri_006-16-2023_20:10:05_0.763774618976614_06-16-2023_20:10:07.png": "gpu0_b_voree_vbtmt_otym_t_trmrtr_gmeritm_wx_n_rmgri_006-16-2023_20:11:22_0.763774618976614_06-16-2023_20:11:24.png",
            },
        )
