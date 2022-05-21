#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import mobile_cv.arch.utils.quantize_utils as qu  # noqa
import torch
import torch.ao.quantization.quantize_fx as quantize_fx


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


class TestFBNetV2Quantize(unittest.TestCase):
    def test_post_quant(self):
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
                    ("ir_k3_sehsig", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        model = torch.ao.quantization.QuantWrapper(model)
        model = fuse_utils.fuse_model(model, inplace=False)

        print(f"Fused model {model}")

        model.qconfig = torch.ao.quantization.default_qconfig
        print(model.qconfig)
        torch.ao.quantization.prepare(model, inplace=True)

        # calibration
        for _ in range(5):
            data = torch.rand([2, 3, 8, 8])
            model(data)

        # Convert to quantized model
        quant_model = torch.ao.quantization.convert(model, inplace=False)
        print(f"Quant model {quant_model}")

        # Run quantized model
        quant_output = quant_model(torch.rand([2, 3, 8, 8]))
        self.assertEqual(quant_output.shape, torch.Size([2, 8, 2, 2]))

        # Trace quantized model
        jit_model = torch.jit.trace(quant_model, data)
        jit_quant_output = jit_model(torch.rand([2, 3, 8, 8]))
        self.assertEqual(jit_quant_output.shape, torch.Size([2, 8, 2, 2]))

    def test_qat(self):
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
                    ("ir_k3_sehsig", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        model.train()

        qconfig_dict = {"": torch.ao.quantization.get_default_qat_qconfig("fbgemm")}
        example_inputs = (torch.rand(2, 3, 8, 8),)
        model_prepared = quantize_fx.prepare_qat_fx(
            model, qconfig_dict, example_inputs=example_inputs
        )

        print(f"Prepared model {model_prepared}")

        # calibration
        for _ in range(5):
            data = torch.rand([2, 3, 8, 8])
            model(data)

        model_prepared.eval()
        model_quant = quantize_fx.convert_fx(model_prepared)

        print(f"Quantized model {model_quant}")

        # # Run quantized model
        quant_output = model_quant(torch.rand([2, 3, 8, 8]))
        self.assertEqual(quant_output.shape, torch.Size([2, 8, 2, 2]))

        # Trace quantized model
        jit_model = torch.jit.trace(model_quant, data)
        jit_quant_output = jit_model(torch.rand([2, 3, 8, 8]))
        self.assertEqual(jit_quant_output.shape, torch.Size([2, 8, 2, 2]))

    def test_qat_lstm(self):
        import torch.nn.quantizable as nnqa

        model = torch.nn.Sequential(
            torch.quantization.QuantStub(),
            torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=2),
            # torch.quantization.DeQuantStub()
        )

        from torch.ao.quantization.quantization_mappings import (
            get_default_qat_module_mappings,
        )

        model.train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        model_prepared = torch.ao.quantization.prepare_qat(
            model,
            mapping={torch.nn.LSTM: nnqa.LSTM, **get_default_qat_module_mappings()},
        )
        print(f"Prepared model {model_prepared}")

        # calibration
        for _ in range(5):
            data = torch.rand([4, 2, 10])
            model(data)

        model_prepared.eval()
        model_quant = torch.ao.quantization.convert(model_prepared)

        print(f"Quantized model {model_quant}")

        # # Run quantized model
        quant_output, (hn, cn) = model_quant(torch.rand([4, 2, 10]))
        self.assertEqual(quant_output.shape, torch.Size([4, 2, 20]))

        # Trace quantized model
        jit_model = torch.jit.trace(model_quant, data)
        jit_quant_output, jit_hcn = jit_model(torch.rand([4, 2, 10]))
        self.assertEqual(jit_quant_output.shape, torch.Size([4, 2, 20]))
