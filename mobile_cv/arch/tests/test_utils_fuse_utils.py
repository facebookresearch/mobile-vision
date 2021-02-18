#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import typing
import unittest

import numpy as np
import torch

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.utils.fuse_utils as fuse_utils
from mobile_cv.arch.layers.batch_norm import (
    NaiveSyncBatchNorm,
    NaiveSyncBatchNorm1d,
    NaiveSyncBatchNorm3d,
)

from . import helper


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


def _find_modules(model, module_to_check):
    for x in model.modules():
        if isinstance(x, module_to_check):
            return True
    return False


class ModelWithRef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cbr = bb.ConvBNRelu(3, 6, kernel_size=3, bn_args="bn")
        self.cbr = cbr
        self.cbr_list = [cbr]

    def forward(self, x):
        return self.cbr(x)


class TestUtilsFuseUtils(unittest.TestCase):
    def test_fuse_convbnrelu(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        ).eval()
        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=False)

        self.assertTrue(_find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused, torch.nn.BatchNorm2d))

        input_size = [2, 3, 7, 7]
        run_and_compare(cbr, fused, input_size)

    def test_fuse_convbnrelu_inplace(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        ).eval()
        self.assertTrue(_find_modules(cbr, torch.nn.BatchNorm2d))

        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=True)

        self.assertFalse(_find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused, torch.nn.BatchNorm2d))

        input_size = [2, 3, 7, 7]
        run_and_compare(cbr, fused, input_size)

    def test_fuse_convnormact(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        )
        cbr = bb.ConvNormAct(cbr.conv, cbr.bn, cbr.relu).eval()
        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=False)

        self.assertTrue(_find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused, torch.nn.BatchNorm2d))

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

        model = _build_model(arch_def, dim_in=3)
        fused_model = fuse_utils.fuse_model(model, inplace=False)
        print(model)
        print(fused_model)

        self.assertTrue(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))

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

        self.assertFalse(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))

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

        model = _build_model(arch_def, dim_in=3)
        fused_model = fuse_utils.fuse_model(model, inplace=False)
        print(model)
        print(fused_model)

        self.assertTrue(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))
        self.assertTrue(_find_modules(fused_model, bb.Swish))

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

        fused_model = fuse_utils.fuse_model(model, inplace=False)
        print(model)
        print(fused_model)

        self.assertTrue(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))
        self.assertTrue(_find_modules(fused_model, torch.nn.ReLU))

        input_size = [1, 3, 4, 4]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_model_with_reference(self):
        model = ModelWithRef()
        model.eval()
        self.assertEqual(id(model.cbr), id(model.cbr_list[0]))

        # keep the same reference after copying
        model_copy = copy.deepcopy(model)
        self.assertEqual(id(model_copy.cbr), id(model_copy.cbr_list[0]))

        fused_model = fuse_utils.fuse_model(model, inplace=False)
        self.assertEqual(id(fused_model.cbr), id(fused_model.cbr_list[0]))
        print(model)
        print(fused_model)

        self.assertTrue(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))

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
        }
        for name, test_args in TEST_MAPS.items():
            with self.subTest(name=name):
                print(f"Testing {name}")
                _test_fuse_conv(self, **test_args)

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
            with self.subTest(name=name):
                print(f"Testing {name}")
                _test_fuse_conv(self, **test_args)


def _test_fuse_conv(
    self, conv_args, bn_args, relu_args, bn_class, input_size, device="cpu"
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

    fused = fuse_utils.fuse_convbnrelu(cbr, inplace=False)

    self.assertTrue(_find_modules(cbr, bn_class))
    self.assertFalse(_find_modules(fused, bn_class))

    run_and_compare(cbr, fused, input_size, device)
