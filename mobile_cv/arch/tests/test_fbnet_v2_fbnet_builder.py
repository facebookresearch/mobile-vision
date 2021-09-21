#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as basic_blocks
import mobile_cv.arch.fbnet_v2.blocks_factory as blocks_factory
import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.fbnet_v2.irf_block as irf_block
import torch


def _build_model(arch_def, dim_in, **kwargs):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    builder.add_basic_args(**arch_def.get("basic_args", {}))
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in, **kwargs)
    model.eval()
    return model, builder


def _get_input(n, c, h, w):
    nchw = n * c * h * w
    input = (torch.arange(nchw, dtype=torch.float32) - (nchw / 2.0)) / (nchw)
    input = input.reshape(n, c, h, w)
    return input


class TestFBNetBuilder(unittest.TestCase):
    def test_fbnet_builder_check_output(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model, builder = _build_model(arch_def, dim_in=3)

        print(model)

        input = _get_input(2, 3, 8, 8)
        output = model(input)
        self.assertEqual(output.shape, torch.Size([2, 8, 2, 2]))

    def test_fbnet_builder_width_divisor(self):
        e6 = {"expansion": 6}

        WIDTH_DIVISOR = 8

        class ConvCheck(torch.nn.Module):
            def __init__(
                self, in_channels, out_channels, *, check_out_channels, **kwargs
            ):
                super().__init__()
                assert kwargs["width_divisor"] == WIDTH_DIVISOR
                self.check_out_channels = check_out_channels
                self.conv = basic_blocks.ConvBNRelu(in_channels, out_channels, **kwargs)

            def forward(self, x):
                ret = self.conv(x)
                assert ret.shape[1] == self.check_out_channels
                return ret

        class IRFCheck(torch.nn.Module):
            def __init__(
                self, in_channels, out_channels, *, check_out_channels, **kwargs
            ):
                super().__init__()
                assert kwargs["width_divisor"] == WIDTH_DIVISOR
                self.check_out_channels = check_out_channels
                self.op = irf_block.IRFBlock(in_channels, out_channels, **kwargs)

            def forward(self, x):
                ret = self.op(x)
                assert ret.shape[1] == self.check_out_channels
                return ret

        blocks_factory.PRIMITIVES.register_dict(
            {
                "conv_check": lambda in_channels, out_channels, stride, **kwargs: ConvCheck(  # noqa
                    in_channels, out_channels, stride=stride, **kwargs
                ),
                "irf_check": lambda in_channels, out_channels, stride, **kwargs: IRFCheck(  # noqa
                    in_channels, out_channels, stride=stride, **kwargs
                ),
            }
        )

        arch_def = {
            "basic_args": {"width_divisor": WIDTH_DIVISOR},
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [
                    (
                        "conv_check",
                        3,
                        2,
                        1,
                        {"check_out_channels": 3, "kernel_size": 3},
                    )
                ],
                # stage 1
                [
                    (
                        "irf_check",
                        8,
                        2,
                        2,
                        e6,
                        {"check_out_channels": 8, "kernel_size": 3},
                    ),
                    (
                        "irf_check",
                        1,
                        1,
                        1,
                        e6,
                        {"check_out_channels": 1, "kernel_size": 5},
                    ),
                ],
            ],
        }
        model, builder = _build_model(arch_def, dim_in=3)
        self.assertEqual(builder.width_divisor, WIDTH_DIVISOR)

        print(model)

        input = _get_input(1, 3, 8, 8)
        output = model(input)
        self.assertEqual(output.shape, torch.Size([1, 1, 2, 2]))

    def test_fbnet_builder_specify_in_channels(self):
        e6 = {"expansion": 6}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # specify in_channels, useful in some cases
                [("conv_k3", 4, 2, 1, {"in_channels": 5})],
                [
                    ("ir_k3", 8, 2, 2, e6),
                ],
            ]
        }

        # dim_in will be overridden by the arch def
        model, builder = _build_model(arch_def, dim_in=3)

        print(model)

        input = _get_input(2, 5, 8, 8)
        output = model(input)
        self.assertEqual(output.shape, torch.Size([2, 8, 2, 2]))

    def test_fbnet_builder_missing_values(self):
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # None will be replaced by the values passed to the builder in 'override_missing'
                [("conv_k3", 4, 2, 1)],
                [
                    ("conv_k1", 8, None, 1),
                    ("conv_k5", None, 2, 1),
                    ("conv_k1", None, 2, 1),
                ],
            ]
        }

        # dim_in will be overridden by the arch def
        model, builder = _build_model(
            arch_def, dim_in=3, override_missing={"out_channels": 2, "stride": 2}
        )

        print(model)

        input = _get_input(2, 3, 8, 8)
        output = model(input)
        self.assertEqual(output.shape, torch.Size([2, 2, 1, 1]))

    def test_fbnet_builder_replace_strs(self):
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                [("conv_k3", 4, 2, 1)],
                [
                    ("conv_k1", 8, "{stride_c1}", 1),
                    ("conv_k5", "{out_channels_c2}", 2, 1),
                    ("conv_k1", None, 2, 1),
                ],
            ]
        }

        # dim_in will be overridden by the arch def
        model, builder = _build_model(
            arch_def,
            dim_in=3,
            override_missing={"out_channels": 2},
            replace_strs={"stride_c1": 2, "out_channels_c2": 4},
        )

        print(model)

        input = _get_input(2, 3, 8, 8)
        output = model(input)
        self.assertEqual(output.shape, torch.Size([2, 2, 1, 1]))


if __name__ == "__main__":
    unittest.main()
