#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
import torch.nn as nn


class EmptyJitScriptModule(torch.jit.ScriptModule):
    def forward(self, x):
        return x


# pyre-fixme[11]: Annotation `Sequential` is not defined as a type.
class M1(nn.Sequential):
    pass


class M2(nn.Sequential):
    pass


def create_model2():
    ret = nn.Sequential(
        nn.Conv2d(
            3,
            8,
            3,
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        ),
        nn.ReLU(),
        nn.Conv2d(
            8,
            8,
            3,
            stride=(2, 2),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        ),
        nn.ConvTranspose2d(
            8,
            4,
            3,
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        ),
        EmptyJitScriptModule(),
        nn.ReLU(),
    )
    ret.OP_INPUT = [2, 3, 16, 16]
    return ret


class TestFlopsEstimation(unittest.TestCase):
    def test_get_flops(self):
        model = create_model2()
        fest = flops_utils.FlopsEstimation(model)

        input = torch.zeros(model.OP_INPUT)
        with fest.enable():
            model(input)
            nparams, nflops = fest.get_flops()

        self.assertAlmostEqual(nparams, 0.00108)
        self.assertAlmostEqual(nflops, 0.221184)

    def test_print_shape(self):
        model = create_model2()
        input = torch.zeros(model.OP_INPUT)
        model_str_before = str(model)
        fest = flops_utils.FlopsEstimation(model)
        model_str_after = str(model)
        # make sure str was not affected when it is not enabled
        self.assertEqual(model_str_before, model_str_after)

        with fest.enable():
            model(input)
            fest.add_flops_info()
            model_str = str(model)
            print(model_str)

        GT_SHAPES_STRS = [
            "input_shapes=[[2, 3, 16, 16]], output_shapes=[2, 8, 16, 16]",
            "input_shapes=[[2, 8, 8, 8]], output_shapes=[2, 4, 8, 8]",
            "ReLU(input_shapes=[[2, 4, 8, 8]], output_shapes=[2, 4, 8, 8], nparams=0.0, nflops=0.0)",  # noqa
            "nparams=0.00108, nflops=0.221184",
        ]
        for x in GT_SHAPES_STRS:
            self.assertIn(x, model_str)

        # make sure the additional informaiton are cleaned up
        model_str_clean = str(model)
        GT_SHAPES = ["input_shapes", "output_shapes", "nparams", "nflops"]
        for x in GT_SHAPES:
            self.assertNotIn(x, model_str_clean)

    def test_callback(self):
        model = create_model2()
        input = torch.zeros(model.OP_INPUT)

        fest = flops_utils.FlopsEstimation(model)

        count = 0
        flops = []

        def flops_callback(fest, model, model_data):
            nonlocal count
            nparams, nflops = fest.get_flops()
            flops.append({"nparams": nparams, "nflops": nflops})
            if count >= 2:
                fest.set_enable(False)
            count += 1

        fest.set_callback(flops_callback)

        fest.set_enable(True)
        for _ in range(5):
            model(input)

        gt_flops = [{"nparams": 0.00108, "nflops": 0.221184}] * 3
        self.assertEqual(gt_flops, flops)

    def test_get_unique_parent_types(self):
        types = [M1, M2, nn.Sequential, M1, nn.Conv2d]
        gt_types = [nn.Sequential, nn.Conv2d]
        unique_types = flops_utils.get_unique_parent_types(types)
        self.assertEqual(unique_types, gt_types)

    def test_duplicated(self):
        """Make sure handles subclasses propertly for mock"""
        model = nn.Sequential(
            M1(), M2(), nn.Conv2d(3, 4, 3), nn.ConvTranspose2d(4, 4, 3), M1()
        )
        input = torch.zeros([1, 3, 4, 4])

        fest = flops_utils.FlopsEstimation(model)

        count = 0
        flops = []

        def flops_callback(fest, model, model_data):
            nonlocal count
            nparams, nflops = fest.get_flops()
            flops.append({"nparams": nparams, "nflops": nflops})
            if count >= 2:
                fest.set_enable(False)
            count += 1

        fest.set_callback(flops_callback)

        fest.set_enable(True)
        for _ in range(5):
            model(input)

        gt_flops = [{"nparams": 0.000252, "nflops": 0.002736}] * 3
        self.assertEqual(gt_flops, flops)

    def test_print_model_shape_nokwargs(self):
        """When print_model_flops are used, all the modules inside the model
        needs to pass the arguments through positional arguments.
        Passing arguments through key values are not supported due to the
        limitation of torch.nn.Module.register_forward_hook() (only the positional
        arguments are given to the module).
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_forward_hook#torch.nn.Module.register_forward_hook # noqa
        """
        src_l = 3
        bsz = 5
        embed_size = 8
        nhead = 2
        query = torch.rand((src_l, bsz, embed_size))
        key = torch.rand((src_l, bsz, embed_size))
        value = torch.rand((src_l, bsz, embed_size))

        class Test(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = torch.nn.MultiheadAttention(embed_size, nhead, dropout=0.1)

            def forward(self, use_kwargs):
                if use_kwargs:
                    # the code will fail as `value` will not be passed to the
                    # hook to compute the flops
                    return self.attn(query, key, value=value)
                return self.attn(query, key, value)

        model = Test()

        # use `use_kwargs == False` will work correctly
        ops = flops_utils.print_model_flops(model, (False,))
        self.assertEqual(len(ops), 2)

        # use `use_kwargs == True` will fail
        with self.assertRaises(Exception) as context:
            flops_utils.print_model_flops(model, (True,))
        self.assertTrue(
            "not enough values to unpack (expected 3, got 2)" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
