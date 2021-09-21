#!/usr/bin/env python3
# buck test mobile-vision/mobile_cv/mobile_cv/arch/tests:test_fbnet_hr

import copy
import random
import unittest
from typing import NamedTuple

import mobile_cv.arch.fbnet_v2.fbnet_hr_modeldef as fbnet_hr_modeldef
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st
from mobile_cv.arch.fbnet_v2.fbnet_builder import FBNetBuilder
from mobile_cv.arch.fbnet_v2.fbnet_hr import (
    FBNetHRBuilder,
    FBNetHRMultiView,
    ViewsToBatch,
)


def _rand_input(n=None, c=None, h=None, w=None):
    if n is None:
        n = random.randint(1, 5)
    if c is None:
        c = random.randint(1, 5)
    if h is None:
        h = random.randrange(32, 129, 32)
    if w is None:
        w = random.randrange(32, 129, 32)
    return torch.randn(n, c, h, w)


class TestModel(torch.nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, k, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, k, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        return y + self.conv2(y)


class DummyStage(NamedTuple):
    out_channels: int


def compare_model_outputs(model_A, model_B, data_shape):
    """Checks whether the same input results in the same output"""
    input_data = torch.rand(*data_shape)
    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        output_A = model_A(input_data)
        output_B = model_B(input_data)
    torch.testing.assert_allclose(output_A, output_B)


class TestFBNetHR(unittest.TestCase):
    def test_build_and_run_model(self):
        """Check that model can run on random input"""
        input_channels = random.randint(1, 5)
        arch_def = {
            "stages": [[("conv", random.randint(2, 6), 1, 1)]]
            * random.randrange(3, 16, 3)
        }
        builder = FBNetBuilder()
        builder.last_depth = input_channels
        builder_hr = FBNetHRBuilder(builder)
        model = builder_hr.build_model(arch_def)
        input = _rand_input(c=input_channels)
        output = model(input)
        gt_output_shape = list(input.shape)
        gt_output_shape[1] = arch_def["stages"][2][0][1]
        self.assertEqual(output.shape, torch.Size(gt_output_shape))

    def test_vs_manual(self):
        """Compare the output of the model with a hand created model

        The manual model is:
            output = conv1(input) + conv2(conv1(input))
        """
        in_c = random.randint(1, 5)
        out_c = random.randint(1, 5)

        # build manual
        model_A = TestModel(in_c, out_c, 3)

        # build fbnet_hr
        builder = FBNetBuilder()
        builder.add_basic_args(bn_args=None, relu_args=None)
        builder.last_depth = in_c
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {
            "stages": [
                [("conv_k3", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("conv_k3", out_c, 1, 1)],
            ]
        }
        model_B = builder_hr.build_model(arch_def)

        # sync models
        model_B.stages[0][0].xif0_0.conv.weight.data = model_A.conv1.weight.clone()
        model_B.stages[2][1].xif5_0.conv.weight.data = model_A.conv2.weight.clone()
        compare_model_outputs(model_A, model_B, _rand_input(c=in_c).shape)

    def test_high_res_channel_input(self):
        """Check high_res option in calculating stage input

        The input channels of stages[i+1][j] should be the output channels of
        stages[i][j] (high res)

        stages[i][j]         stages[i][j+1]
              |                   |
           combiner ---------------
              |
        stages[i+1][j]
        """
        builder = FBNetBuilder()
        builder.add_basic_args(bn_args=None, relu_args=None)
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {
            "stages": [
                [("skip", 3, 1, 1)],
                [("skip", 3, 1, 1)],
                [("conv_k3", 1, 1, 1)],
                # stages[i][j+1]
                [("conv_k3", 2, 1, 1)],
                [("skip", 2, 1, 1)],
                [("conv_k3", 3, 1, 1)],
            ],
            "stage_combiners": ["add"],
            "combiner_path": "high_res",
        }
        model = builder_hr.build_model(arch_def, input_channels=3)
        model(torch.randn(1, 3, 32, 32))

    def test_low_res_channel_input(self):
        """Check low_res option in calculating stage input

        The input channels of stages[i+1][j] should be the output channels of
        stages[i][j+1] (low res)

        stages[i][j]         stages[i][j+1]
              |                   |
           combiner ---------------
              |
        stages[i+1][j]
        """
        builder = FBNetBuilder()
        builder.add_basic_args(bn_args=None, relu_args=None)
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {
            "stages": [
                [("skip", 3, 1, 1)],
                [("skip", 3, 1, 1)],
                [("conv_k3", 1, 1, 1)],
                # stages[i][j+1]
                [("conv_k3", 2, 1, 1)],
                [("skip", 2, 1, 1)],
                [("conv_k3", 3, 1, 1)],
            ],
            "stage_combiners": ["add"],
            "combiner_path": "low_res",
        }
        model = builder_hr.build_model(arch_def, input_channels=3)
        model(torch.randn(1, 3, 32, 32))

    def test_width_ratio(self):
        """Check width_ratio > 1.0"""
        builder = FBNetBuilder(width_ratio=2.2)
        builder.add_basic_args(dw_skip_bnrelu=True, zero_last_bn_gamma=False)
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {
            "stages": [
                # [op, c, s, n, ...]
                # downsampled (x2)
                [("ir_k3", 16, 2, 1)],
                [("skip", 16, 1, 1)],
                [("ir_k3", 8, -2, 1)],
                # downsampled (x4)
                [("ir_k3", 24, 2, 2)],
                [("skip", 24, 1, 1)],
                [("ir_k3", 16, -2, 1)],
                # downsampled (x8)
                [("ir_k3", 32, 2, 3)],
                [("skip", 32, 1, 1)],
                [("ir_k3", 24, -2, 1)],
                # downsampled (x16)
                [("ir_k3", 40, 2, 6)],
                [("skip", 40, 1, 1)],
                [("ir_k3", 32, -2, 1)],
            ]
        }
        model = builder_hr.build_model(arch_def, input_channels=3)
        model(torch.randn(1, 3, 64, 64))

    def test_customize_width_ratio(self):
        """Check customize block_wise width_ratio"""
        builder = FBNetBuilder(width_ratio=2.0)
        builder.add_basic_args(dw_skip_bnrelu=True, zero_last_bn_gamma=False)
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {
            "stages": [
                # [op, c, s, n, ...]
                # downsampled (x2)
                [("ir_k3", 16, 2, 1)],
                [("skip", 16, 1, 1)],
                [("ir_k3", 8, -2, 1, {"width_ratio": 1.0})],
                # downsampled (x4)
                [("ir_k3", 24, 2, 2)],
                [("skip", 24, 1, 1)],
                [("ir_k3", 16, -2, 1)],
                # downsampled (x8)
                [("ir_k3", 32, 2, 3)],
                [("skip", 32, 1, 1)],
                [("ir_k3", 24, -2, 1)],
                # downsampled (x16)
                [("ir_k3", 40, 2, 6)],
                [("skip", 40, 1, 1)],
                [("ir_k3", 32, -2, 1)],
            ]
        }
        model = builder_hr.build_model(arch_def, input_channels=3)

        import mobile_cv.lut.lib.pt.flops_utils as flops_utils

        out = flops_utils.print_model_flops(model, [torch.randn(1, 3, 64, 64)])

        self.assertEqual(out.size(1), 8)

    def test_stage_combine_concat(self):
        """Check concat as stage_combiner"""
        builder = FBNetBuilder()
        builder.add_basic_args(bn_args=None, relu_args=None)
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {
            "stages": [
                # [op, c, s, n, ...]
                # original res
                [("conv_k1", 8, 1, 1)],
                [("skip", 8, 1, 1)],
                [("conv_k1", 1, 1, 1, {"in_channels": 16})],
                # downsampled (x2)
                [("conv_k1", 16, 2, 1)],
                [("skip", 16, 1, 1)],
                [("conv_k1", 8, -2, 1)],
            ],
            "stage_combiners": [
                # original res
                "concat"
            ],
        }
        model = builder_hr.build_model(arch_def, input_channels=3)
        out = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(out.shape, torch.Size([1, 1, 32, 32]))

    def test_out_channels(self):
        """Check if we set the out_channels properly"""
        builder = FBNetBuilder()
        builder.add_basic_args(dw_skip_bnrelu=True, zero_last_bn_gamma=False)
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {
            "stages": [
                # [op, c, s, n, ...]
                # downsampled (x2)
                [("ir_k3", 16, 2, 1)],
                [("skip", 16, 1, 1)],
                [("ir_k3", 8, -2, 1)],
                # downsampled (x4)
                [("ir_k3", 24, 2, 2)],
                [("skip", 24, 1, 1)],
                [("ir_k3", 16, -2, 1)],
                # downsampled (x8)
                [("ir_k3", 32, 2, 3)],
                [("skip", 32, 1, 1)],
                [("ir_k3", 24, -2, 1)],
                # downsampled (x16)
                [("ir_k3", 40, 2, 6)],
                [("skip", 40, 1, 1)],
                [("ir_k3", 32, -2, 1)],
            ]
        }
        model = builder_hr.build_model(arch_def, input_channels=3)
        self.assertEqual(model.out_channels, 8)


class TestModelDef(unittest.TestCase):
    def test_run_modeldef(self):
        """Check that modeldefs can be run"""
        arch_defs = fbnet_hr_modeldef.MODEL_ARCH
        self.assertTrue(len(arch_defs) > 0)
        for _, arch_def in arch_defs.items():
            builder = FBNetBuilder()
            builder.last_depth = 1
            builder_hr = FBNetHRBuilder(builder)
            model = builder_hr.build_model(arch_def)
            model.eval()
            input_data = torch.rand(1, 1, 32, 32)
            model(input_data)

    def test_modeldef_default(self):
        """Check that the TestModel modeldef is the same as TestModel"""
        in_c = random.randint(1, 5)
        out_c = 16

        # build manual
        model_A = TestModel(in_c, out_c, 3)

        # build fbnet_hr
        builder = FBNetBuilder()
        builder.add_basic_args(bn_args=None, relu_args=None)
        builder.last_depth = in_c
        builder_hr = FBNetHRBuilder(builder)
        arch_def = fbnet_hr_modeldef.MODEL_ARCH["TestModel"]
        model_B = builder_hr.build_model(arch_def)

        # sync models
        model_B.stages[0][0].xif0_0.conv.weight.data = model_A.conv1.weight.clone()
        model_B.stages[2][1].xif5_0.conv.weight.data = model_A.conv2.weight.clone()
        compare_model_outputs(model_A, model_B, _rand_input(c=in_c).shape)


class TestFBNetHRMultiView(unittest.TestCase):
    def _get_multiview_config(
        self, v=2, corr_type="unfold", k=0, d=10, s1=1, s2=2, selected_layers=None
    ):
        if selected_layers is None:
            selected_layers = [0]
        return {
            "nviews": v,
            "correlation": {
                "corr_type": corr_type,
                "k": k,
                "d": d,
                "s1": s1,
                "s2": s2,
                "selected_layers": selected_layers,
            },
        }

    def test_views_to_batch(self):
        """Check n, v, c, h, w converted to nv, c, h, w"""
        n, v, c, h, w = [random.randint(2, 10) for _ in range(5)]
        input = torch.randn(n, v, c, h, w)
        block = ViewsToBatch(v)
        output = block(copy.deepcopy(input))
        self.assertEqual(output.shape, torch.Size([n * v, c, h, w]))
        # check the new batch is view0_n0, view1_n0
        torch.testing.assert_allclose(input[0, 0, :], output[0, :])
        torch.testing.assert_allclose(input[0, 1, :], output[1, :])

    def test_views_to_batch_in_model(self):
        """Check n, v, c, h, w converted to nv, c, h, w"""
        n, v, c, h, w = [random.randint(2, 10) for _ in range(5)]
        config = self._get_multiview_config(v)
        block = FBNetHRMultiView([[DummyStage(c)]], [], config)
        input = torch.randn(n, v, c, h, w)
        output = block.views_to_batch(copy.deepcopy(input))
        self.assertEqual(output.shape, torch.Size([n * v, c, h, w]))
        # check the new batch is view0_n0, view1_n0
        torch.testing.assert_allclose(input[0, 0, :], output[0, :])
        torch.testing.assert_allclose(input[0, 1, :], output[1, :])

    def test_select_view(self):
        """Check view can be selected out of nv, c, h, w"""
        n, v, c, h, w = [random.randint(2, 10) for _ in range(5)]
        view = random.randint(0, v - 1)
        config = self._get_multiview_config(v)
        block = FBNetHRMultiView([[DummyStage(c)]], [], config)
        input = torch.randn(n * v, c, h, w)
        output = block.select_view(input, view=view)
        self.assertEqual(output.shape, torch.Size([n, c, h, w]))
        # check returned is view_n0, view_n1
        for _n in range(n):
            torch.testing.assert_allclose(input[_n * v + view, :], output[_n, :])

    def test_projection_shape(self):
        """Check projection block output shape"""
        n, v, ref_c, h, w, d = [random.randint(2, 10) for _ in range(6)]
        config = self._get_multiview_config(v, d=d)
        corr_c = (v - 1) * (2 * d // config["correlation"]["s2"] + 1) ** 2

        block = FBNetHRMultiView([[DummyStage(ref_c)]], [], config)
        input = torch.randn(n, ref_c + corr_c, h, w)
        assert len(block.stereo_projections) == 1, block.stereo_projections.keys()
        (projection,) = block.stereo_projections.values()
        output = projection(input)
        self.assertEqual(output.shape, torch.Size([n, ref_c, h, w]))

        # zero initialization means output should be the same as input ref_c
        torch.testing.assert_allclose(input[:, :ref_c, :], output)

    def test_projection_zero_init(self):
        """Check projection block zero init"""
        n, v, ref_c, h, w, d = [random.randint(2, 10) for _ in range(6)]
        config = self._get_multiview_config(v, d=d)
        corr_c = (v - 1) * (2 * d // config["correlation"]["s2"] + 1) ** 2

        block = FBNetHRMultiView([[DummyStage(ref_c)]], [], config)
        input = torch.randn(n, ref_c + corr_c, h, w)
        assert len(block.stereo_projections) == 1, block.stereo_projections.keys()
        (projection,) = block.stereo_projections.values()
        output = projection(input)

        # zero initialization means output should be the same as input ref_c
        torch.testing.assert_allclose(input[:, :ref_c, :], output)

    def test_correlation_shape(self):
        """Check correlation block output shape

        Reduce the size of the data bc naive correlation is slow
        """
        n, ref_c, h, w, d = [random.randint(2, 5) for _ in range(5)]
        v = 2
        config = self._get_multiview_config(v, d=d)
        corr_c = (v - 1) * (2 * d // config["correlation"]["s2"] + 1) ** 2

        block = FBNetHRMultiView([[DummyStage(ref_c)]], [], config)
        input1 = torch.randn(n, ref_c, h, w)
        input2 = torch.randn(n, ref_c, h, w)
        input1_alpha = torch.ones_like(input1[:, 0:1]).type(torch.bool)
        input2_alpha = torch.ones_like(input2[:, 0:1]).type(torch.bool)
        output = block.correlations[0](input1, input2, input1_alpha, input2_alpha)
        self.assertEqual(output.shape, torch.Size([n, corr_c, h, w]))

    def test_build_correlation_model(self):
        """Check model is multivew fbnet

        If the model only has correlation and projection is zero init,
        then the output will be the same as the ref view
        """
        n, h, w, d, v = [random.randint(2, 5) for _ in range(5)]
        c = 1
        arch_def = {"stages": [[("skip", 1, 1, 1)]] * 3}
        arch_def.update(self._get_multiview_config(v, d=d))
        builder = FBNetBuilder()
        builder_hr = FBNetHRBuilder(builder)
        model = builder_hr.build_model(arch_def, input_channels=c)
        self.assertTrue(isinstance(model, FBNetHRMultiView))

        # check output shape
        input = torch.randn(n, v, c + 1, h, w)
        stereo_examples = torch.ones(n, dtype=torch.bool)
        output = model(input, stereo_examples)
        self.assertEqual(output.shape, torch.Size([n, 1, h, w]))

        # zero init means the output is the same as the ref view
        torch.testing.assert_allclose(input[:, 0, :-1], output)

    def test_conv_model(self):
        """Check building the model with convs"""
        n, c, h, w, d, v = [random.randint(2, 5) for _ in range(6)]
        h = 2 * h
        w = 2 * w
        arch_def = {"stages": [[("conv", 1, 1, 1)]] * 6}
        arch_def.update(self._get_multiview_config(v, d=d))
        builder = FBNetBuilder()
        builder_hr = FBNetHRBuilder(builder)
        model = builder_hr.build_model(arch_def, input_channels=c)
        self.assertTrue(isinstance(model, FBNetHRMultiView))

        # check output shape
        input = torch.randn(n, v, c + 1, h, w)
        stereo_examples = torch.ones(n, dtype=torch.bool)
        output = model(input, stereo_examples)
        self.assertEqual(output.shape, torch.Size([n, 1, h, w]))

    @given(
        n=st.integers(2, 5),
        c=st.integers(2, 5),
        h=st.integers(2, 5),
        w=st.integers(2, 5),
        d=st.integers(2, 5),
        v=st.integers(2, 5),
        arch_channels=st.integers(2, 6),
        arch_stages=st.integers(1, 5),
    )
    @settings(max_examples=20, deadline=None)
    def test_load_fbnethr(self, n, c, h, w, d, v, arch_channels, arch_stages):
        """Check load fbnethr weights into model

        The outputs should be the same as the multivew model projection
        initialized to zero in correlation channels
        """
        h = 4 * h
        w = 4 * w
        arch_stages = 3 * arch_stages  # FBNetHR requires multiple of 3

        # build fbnet_hr
        builder = FBNetBuilder()
        builder_hr = FBNetHRBuilder(builder)
        arch_def = {"stages": [[("conv", arch_channels, 1, 1)]] * arch_stages}
        model_A = builder_hr.build_model(arch_def, c)

        # load state_dict into fbnet multiview model
        arch_def.update(self._get_multiview_config(v, d=d))
        model_B = builder_hr.build_model(arch_def, c)
        model_B.load_fbnethr_state_dict(model_A.state_dict())

        # compare models
        input = torch.randn(n, v, c + 1, h, w)
        stereo_examples = torch.ones(n, dtype=torch.bool)
        model_A.eval()
        model_B.eval()
        with torch.no_grad():
            output_A = model_A(input[:, 0, :-1])
            output_B = model_B(input, stereo_examples)
        torch.testing.assert_allclose(output_A, output_B)
