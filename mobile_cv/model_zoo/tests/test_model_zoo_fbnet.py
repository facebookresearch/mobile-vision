#!/usr/bin/env python3

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from mobile_cv.model_zoo.models.fbnet import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess_from_model_info
from utils import is_devserver


class TestModelZooFBNet(unittest.TestCase):
    def test_fbnet(self):
        load_pretrained = is_devserver()
        model = fbnet("fbnet_a", pretrained=load_pretrained)
        data = torch.zeros([1, 3, 224, 224])
        out = model(data)
        self.assertEqual(out.size(), torch.Size([1, 1000]))

    def test_fbnet_arch_def(self):
        model_arch = {
            "block_op_type": {
                "first": "conv_hs",
                "stages": [
                    # stage 0
                    ["ir_k3"],
                    # stage 1
                    ["ir_k3"],
                ],
                "last": "ir_pool_hs",
            },
            "block_cfg": {
                "first": [16, 2],
                "stages": [
                    # [t, c, n, s]
                    # stage 0
                    [[1, 16, 1, 2]],
                    # stage 1
                    [[6, 24, 1, 2]],
                ],
            },
            "last": [6, 1984, 1, 1],
        }

        model = fbnet(model_arch, pretrained=False)
        data = torch.zeros([1, 3, 224, 224])
        out = model(data)
        self.assertEqual(out.size(), torch.Size([1, 1000]))

    def test_fbnet_i8f(self):
        load_pretrained = is_devserver()
        model = fbnet("fbnet_c_i8f", pretrained=load_pretrained)
        data = torch.zeros([1, 3, 224, 224])
        model(data)

    def test_fbnet_flops(self):
        """
        buck run @mode/dev-nosan //mobile-vision/projects/model_zoo/tests:test_model_zoo_fbnet \
        -- test_model_zoo_fbnet.TestModelZooFBNet.test_fbnet_flops
        """

        MODELS = [
            # "fbnet_ase",
            "fbnet_cse",
            # "eff_0",
            "FBNetV2_L1",
            # "FBNetV2_L2",
            # "FBNetV2_L3",
            # "eff_2",
            # "eff_3",
            # "eff_4",
        ]
        load_pretrained = is_devserver()
        for name in MODELS:
            model = fbnet(name, pretrained=load_pretrained)
            res = model.backbone.arch_def.get("input_size", 224)
            inputs = (torch.zeros([1, 3, res, res]),)
            flops_utils.print_model_flops(model, inputs)

    def test_fbnet_with_preprocess(self):
        load_pretrained = is_devserver()
        model = fbnet("fbnet_cse", pretrained=load_pretrained)
        if load_pretrained:
            preprocess = get_preprocess_from_model_info(model.model_info)
            from PIL import Image

            img = Image.new("RGB", (480, 320))
            data = preprocess(img)
            data_batch = data.unsqueeze(0)
        else:
            # preprocess information is only available for pretrained model
            res = model.backbone.arch_def.get("input_size", 224)
            data_batch = torch.zeros([1, 3, res, res])
        out = model(data_batch)
        self.assertEqual(out.size(), torch.Size([1, 1000]))


if __name__ == "__main__":
    unittest.main()
