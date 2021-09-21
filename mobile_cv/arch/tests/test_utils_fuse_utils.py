#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import typing
import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import numpy as np
import torch
import torch.fx
from mobile_cv.arch.layers.batch_norm import (
    NaiveSyncBatchNorm,
    NaiveSyncBatchNorm1d,
    NaiveSyncBatchNorm3d,
)

from . import helper


# needed for the unit test test_fuse_model_fx_with_assert
torch.fx.wrap("len")


def run_and_compare(model_before, model_after, input_size, device="cpu"):
    inputs = torch.zeros(input_size, requires_grad=False).to(device)
    model_before.to(device)
    model_after.to(device)
    output_before = model_before(inputs)
    output_after = model_after(inputs)

    np.testing.assert_allclose(
        output_before.to("cpu").detach(),
        output_after.to("cpu").detach(),
        rtol=0,
        atol=1e-4,
    )


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


class ModelWithRef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cbr = bb.ConvBNRelu(3, 3, kernel_size=3, bn_args="bn")
        self.cbr = cbr
        self.cbr_list = torch.nn.ModuleList([cbr])

    def forward(self, x):
        y = self.cbr(x)
        y = self.cbr_list[0](y)
        return y


class TestUtilsFuseUtils(unittest.TestCase):
    def test_fuse_convbnrelu(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        ).eval()
        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=False)

        self.assertTrue(helper.find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused, torch.nn.BatchNorm2d))

        input_size = [2, 3, 7, 7]
        run_and_compare(cbr, fused, input_size)

    def test_fuse_convbnrelu_inplace(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        ).eval()
        self.assertTrue(helper.find_modules(cbr, torch.nn.BatchNorm2d))

        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=True)

        self.assertFalse(helper.find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused, torch.nn.BatchNorm2d))

        input_size = [2, 3, 7, 7]
        run_and_compare(cbr, fused, input_size)

    def test_fuse_convnormact(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        )
        cbr = bb.ConvNormAct(cbr.conv, cbr.bn, cbr.relu).eval()
        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=False)

        self.assertTrue(helper.find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused, torch.nn.BatchNorm2d))

        input_size = [2, 3, 7, 7]
        run_and_compare(cbr, fused, input_size)

    def test_fuse_model(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        for use_fx in [False, True]:
            with self.subTest(use_fx=use_fx):
                model = _build_model(arch_def, dim_in=3)
                fused_model = fuse_utils.fuse_model(model, inplace=False, use_fx=use_fx)
                print(model)
                print(fused_model)

                self.assertTrue(helper.find_modules(model, torch.nn.BatchNorm2d))
                self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))

                input_size = [2, 3, 8, 8]
                run_and_compare(model, fused_model, input_size)

    def test_fuse_model_inplace(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        fused_model = fuse_utils.fuse_model(model, inplace=True)
        print(model)
        print(fused_model)

        self.assertFalse(helper.find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))

        input_size = [2, 3, 8, 8]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_model_swish(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args, {"relu_args": "swish"})],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        for use_fx in [False, True]:
            with self.subTest(use_fx=use_fx):
                model = _build_model(arch_def, dim_in=3)
                fused_model = fuse_utils.fuse_model(model, inplace=False, use_fx=use_fx)
                print(model)
                print(fused_model)

                self.assertTrue(helper.find_modules(model, torch.nn.BatchNorm2d))
                self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))
                # with fx fusing, bb.Swish() no longer exist
                self.assertTrue(helper.find_modules(fused_model, torch.nn.Sigmoid))

                input_size = [2, 3, 8, 8]
                run_and_compare(model, fused_model, input_size)

    def test_fuse_model_customized(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv = torch.nn.Conv2d(3, 4, 1)
                self.bn = torch.nn.BatchNorm2d(4, 4)
                self.conv2 = torch.nn.Conv2d(4, 2, 1)
                self.bn2 = torch.nn.BatchNorm2d(2, 2)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                return x

        @fuse_utils.FUSE_LIST_GETTER.register(Module)
        def _get_fuser_name_cbr(
            module: torch.nn.Module,
            supported_types: typing.Dict[str, typing.List[torch.nn.Module]],
        ):
            return [["conv", "bn"], ["conv2", "bn2", "relu2"]]

        model = Module().eval()

        fused_model = fuse_utils.fuse_model(model, inplace=False, use_fx=False)
        print(model)
        print(fused_model)

        self.assertTrue(helper.find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))
        self.assertTrue(helper.find_modules(fused_model, torch.nn.ReLU))

        input_size = [1, 3, 4, 4]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_model_fx(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv = torch.nn.Conv2d(3, 4, 1)
                self.bn = torch.nn.BatchNorm2d(4, 4)
                self.conv2 = torch.nn.Conv2d(4, 2, 1)
                self.bn2 = torch.nn.BatchNorm2d(2, 2)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                return x

        model = Module().eval()

        fused_model = fuse_utils.fuse_model_fx(model)
        print(model)
        print(fused_model)

        self.assertTrue(helper.find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))
        self.assertTrue(helper.find_modules(fused_model, torch.nn.ReLU))

        input_size = [1, 3, 4, 4]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_model_fx_with_assert(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv = torch.nn.Conv2d(3, 4, 1)
                self.bn = torch.nn.BatchNorm2d(4, 4)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x_len = len(x.shape)
                torch._assert(x_len == 4, f"{x_len}")
                x = self.bn(x)
                x = self.relu2(x)
                return x

        model = Module().eval()

        fused_model = fuse_utils.fuse_model_fx(model)
        print(model)
        print(fused_model)

        self.assertTrue(helper.find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))
        self.assertTrue(helper.find_modules(fused_model, torch.nn.ReLU))

        input_size = [1, 3, 4, 4]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_model_with_reference(self):
        model = ModelWithRef()
        model.eval()
        self.assertEqual(id(model.cbr), id(model.cbr_list[0]))

        # keep the same reference after copying
        model_copy = copy.deepcopy(model)
        self.assertEqual(id(model_copy.cbr), id(model_copy.cbr_list[0]))

        fused_model = fuse_utils.fuse_model(model, inplace=False, use_fx=False)
        self.assertEqual(id(fused_model.cbr), id(fused_model.cbr_list[0]))
        print(model)
        print(fused_model)

        self.assertTrue(helper.find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))

        input_size = [2, 3, 8, 8]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_convs(self):
        TEST_MAPS = {
            "test_fuse_native_sync_bn": {
                "conv_args": "conv",
                "bn_args": "sync_bn",
                "relu_args": "relu",
                "bn_class": NaiveSyncBatchNorm,
                "input_size": [2, 3, 7, 7],
            },
            "test_fuse_conv3d": {
                "conv_args": "conv3d",
                "bn_args": "bn3d",
                "relu_args": "relu",
                "bn_class": torch.nn.BatchNorm3d,
                "input_size": [2, 3, 8, 7, 7],
            },
            "test_fuse_conv3d_syncbn": {
                "conv_args": "conv3d",
                "bn_args": "naiveSyncBN3d",
                "relu_args": "relu",
                "bn_class": NaiveSyncBatchNorm3d,
                "input_size": [2, 3, 8, 7, 7],
            },
            "test_fuse_conv1d": {
                "conv_args": "conv1d",
                "bn_args": "bn1d",
                "relu_args": "relu",
                "bn_class": torch.nn.BatchNorm1d,
                "input_size": [2, 3, 7],
            },
            "test_fuse_conv1d_syncbn": {
                "conv_args": "conv1d",
                "bn_args": "naiveSyncBN1d",
                "relu_args": "relu",
                "bn_class": NaiveSyncBatchNorm1d,
                "input_size": [2, 3, 7],
            },
            "test_fuse_linear_relu": {
                "conv_args": "linear",
                "bn_args": None,
                "relu_args": "relu",
                "input_size": [2, 3],
            },
        }
        for name, test_args in TEST_MAPS.items():
            for use_fx in [True, False]:
                with self.subTest(name=name, use_fx=use_fx):
                    print(f"Testing {name} with use_fx={use_fx}")
                    _test_fuse_conv(self, use_fx=use_fx, **test_args)

    @helper.skip_if_no_gpu
    @helper.enable_ddp_env
    def test_fuse_convs_cuda(self):
        TEST_MAPS = {
            "test_fuse_conv3d_syncbn_torch": {
                "conv_args": "conv3d",
                "bn_args": "sync_bn_torch",
                "relu_args": "relu",
                "bn_class": torch.nn.SyncBatchNorm,
                "input_size": [2, 3, 8, 7, 7],
                # torch.nn.SyncBatchNorm could only run on GPU
                "device": "cuda",
            }
        }
        for name, test_args in TEST_MAPS.items():
            for use_fx in [True, False]:
                with self.subTest(name=name, use_fx=use_fx):
                    print(f"Testing {name} with use_fx={use_fx}")
                    _test_fuse_conv(self, use_fx=use_fx, **test_args)

    def test_fuse_fx_recursively(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = ModelWithRef()

            def forward(self, x):
                # make the module not symbolic traceable
                if x.sum() > 0:
                    return self.conv(x)
                else:
                    return self.conv(x * 2)

        model = Model().eval()
        fused_model = fuse_utils.fuse_model_fx(model)
        self.assertTrue(helper.find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(helper.find_modules(fused_model, torch.nn.BatchNorm2d))
        self.assertTrue(helper.find_modules(fused_model, torch.nn.Conv2d))

    def test_swap_modules(self):
        class SubConv(torch.nn.Conv2d):
            @classmethod
            def cast(cls, module):
                ret = copy.deepcopy(module)
                ret.__class__ = torch.nn.Conv2d
                return ret

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = SubConv(3, 3, 1)

            def forward(self, x):
                return self.conv(x)

        swap_mapping = {SubConv: lambda x: x.cast(x)}

        model = Model().eval()
        new_model = fuse_utils.swap_modules(model, swap_mapping)
        self.assertTrue(helper.find_modules(model, SubConv))
        self.assertFalse(helper.find_modules(new_model, SubConv))
        self.assertTrue(helper.find_modules(new_model, torch.nn.Conv2d))


def _test_fuse_conv(
    self,
    conv_args,
    bn_args,
    relu_args,
    input_size,
    bn_class=None,
    device="cpu",
    use_fx=False,
):
    cbr = bb.ConvBNRelu(
        3,
        6,
        kernel_size=3,
        padding=1,
        conv_args=conv_args,
        bn_args=bn_args,
        relu_args=relu_args,
    ).to(device)

    for _ in range(3):
        inputs = torch.rand(input_size).to(device)
        cbr(inputs)

    cbr.eval()

    if not use_fx:
        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=False)
    else:
        fused = fuse_utils.fuse_model_fx(cbr)

    print(fused)

    if bn_class is not None:
        self.assertTrue(helper.find_modules(cbr, bn_class))
        self.assertFalse(helper.find_modules(fused, bn_class))
    self.assertTrue(helper.find_modules(fused, type(cbr.relu)))
    self.assertTrue(helper.find_modules(fused, type(cbr.conv)))

    run_and_compare(cbr, fused, input_size, device)
