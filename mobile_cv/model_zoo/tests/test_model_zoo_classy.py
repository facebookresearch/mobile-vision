#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from mobile_cv.model_zoo.models import model_zoo_factory


class TestModelZooClassy(unittest.TestCase):
    def test_fbnet_classy_fbnetv2(self):
        model_cfg = {
            "name": "fbnet_v2",
            "arch": "fbnet_cse",
            "heads": [
                {
                    "name": "cls_conv_head",
                    "unique_id": "head",
                    "in_channels": 1984,
                    "num_classes": 1000,
                    "fork_block": "backbone",
                }
            ],
        }

        model = model_zoo_factory.get_model("classy", model_cfg=model_cfg)
        model.eval()
        with torch.no_grad():
            data = torch.zeros([1, 3, 224, 224])
            out = model(data)
        self.assertEqual(out.size(), torch.Size([1, 1000]))

    def test_fbnet_classy_resnext(self):
        model = model_zoo_factory.get_model("classy_resnext101_32x4d")
        model.eval()
        with torch.no_grad():
            data = torch.zeros([1, 3, 224, 224])
            out = flops_utils.print_model_flops(model, [data])
            self.assertEqual(out.size(), torch.Size([1, 2048, 7, 7]))

    def test_fbnet_classy_resnext_video_audio(self):
        model = model_zoo_factory.get_model("classy_prod_resnext3d_video_audio")
        model.eval()
        with torch.no_grad():
            data = {
                "video": torch.zeros([1, 3, 4, 112, 112]),
                "audio": torch.zeros([1, 1, 100, 40]),
            }
            out = flops_utils.print_model_flops(model, [data])

            self.assertEqual(out["default_head"].size(), torch.Size([1, 11]))
            self.assertEqual(out["visual_head"].size(), torch.Size([1, 11]))
            self.assertEqual(out["audio_head"].size(), torch.Size([1, 11]))

    def test_fbnet_classy_prodresnet3d(self):
        model_cfg = {
            "name": "prod_resnext3d",
            "frames_per_clip": 8,
            "input_planes": 3,
            "clip_crop_size": 112,
            "skip_transformation_type": "postactivated_shortcut",
            "residual_transformation_type": "postactivated_bottleneck_transformation",
            "num_blocks": [3, 4, 6, 3],
            # "input_key": "video",
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 3,
            "stem_spatial_kernel": 7,
            "stem_maxpool": False,
            "stage_planes": 256,
            "stage_temporal_kernel_basis": [[3], [3], [3], [3]],
            "temporal_conv_1x1": [False, False, False, False],
            "stage_temporal_stride": [1, 2, 2, 2],
            "stage_spatial_stride": [1, 2, 2, 2],
            "num_groups": 256,
            "width_per_group": 1,
            "num_classes": 11,
            "zero_init_residual_transform": True,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "pool_size": [1, 1, 1],
                    "activation_func": "softmax",
                    "num_classes": 11,
                    "fork_block": "pathway0-stage4-block2",
                    "in_plane": 2048,
                    "use_dropout": False,
                }
            ],
        }

        model = model_zoo_factory.get_model("classy", model_cfg=model_cfg)
        model.eval()
        with torch.no_grad():
            data = torch.zeros([1, 3, 8, 112, 112])
            out = flops_utils.print_model_flops(model, [data])
        self.assertEqual(out.size(), torch.Size([1, 11]))

    def test_fbnet_classy_fbnet_video_audio(self):
        model_cfg = {
            "name": "fbnet_v2_multimodal",
            # "arch": "eff_2_3d_va",
            # "arch": "vt_3d_va",
            "arch": "vt_3d_va_syncbn",
            "init_channels": {"video": 3, "audio": 1},
            "input_keys": {"video": "video", "audio": "audio"},
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "activation_func": "softmax",
                    "pool_size": [1, 1, 1],
                    "num_classes": 11,
                    "fork_block": "fusion",
                    "in_plane": 256,
                    "use_dropout": False,
                },
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "visual_head",
                    "activation_func": "softmax",
                    "num_classes": 11,
                    "fork_block": "video",
                    "in_plane": 1408,
                    "use_dropout": False,
                },
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "audio_head",
                    "activation_func": "softmax",
                    "num_classes": 11,
                    "fork_block": "audio",
                    "in_plane": 1408,
                    "use_dropout": False,
                },
            ],
        }
        model = model_zoo_factory.get_model("classy", model_cfg=model_cfg)
        model.eval()
        with torch.no_grad():
            data = {
                "video": torch.zeros([1, 3, 4, 112, 112]),
                "audio": torch.zeros([1, 1, 100, 40]),
            }
            out = flops_utils.print_model_flops(model, [data])

            self.assertEqual(out["default_head"].size(), torch.Size([1, 11]))
            self.assertEqual(out["visual_head"].size(), torch.Size([1, 11]))
            self.assertEqual(out["audio_head"].size(), torch.Size([1, 11]))


if __name__ == "__main__":
    unittest.main()
