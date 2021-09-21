#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.fbnet_v2.visual_transformer_3d as vt3d
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import mobile_cv.arch.utils.jit_utils as ju
import mobile_cv.arch.utils.quantize_utils as qu
import torch


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


class TestFBNetV2VisualTransformer(unittest.TestCase):
    def test_post_quant_vt(self):
        arch_def = {
            "blocks": [
                [
                    ["vt3d", 4, 1, 1, {"init_token": True}],
                    ["vt3d", 4, 1, 1],
                    ["vt3d", 4, 1, 1, {"output_mode": "token"}],
                ]
            ]
        }

        model = _build_model(arch_def, dim_in=4)
        model = fuse_utils.fuse_model(model, inplace=False)

        data = torch.rand([2, 4, 8, 7, 7])
        data1 = torch.rand([2, 4, 8, 10, 7])

        data_list = [(data,), (data1,)]

        # VisualTransformer3D could not be scripted as multiple input/output types
        #  are not well supported in torchscript
        # _test_fp32_scriptable(self, model, data_list, use_trace=False)
        # _test_quant_scriptable(self, model, data_list, use_trace=False)

        # needs to convert to a traceable model before tracing
        traceable_model = ju.get_traceable_model(model)
        _test_fp32_scriptable(
            self, model, data_list, use_trace=True, traceable_model=traceable_model
        )
        _test_quant_scriptable(
            self, model, data_list, use_trace=True, traceable_model=traceable_model
        )

    def test_vt_tokenizer3d_init_token_scriptable(self):
        model = vt3d.Tokenizer3D(
            16,
            8,
            4,
            init_token=True,
            head=4,
            norm_layer_1d=vt3d.get_norm("BN", dim=1),
            norm_layer_3d=vt3d.get_norm("BN", dim=3),
        ).eval()
        model = fuse_utils.fuse_model(model, inplace=False)

        feature = torch.rand([2, 4, 8, 7, 7])
        feature1 = torch.rand([2, 4, 8, 10, 7])
        data_list = [(feature, None), (feature1, None)]

        _test_fp32_scriptable(self, model, data_list)
        _test_quant_scriptable(self, model, data_list)

    def test_vt_tokenizer3d_has_token_scriptable(self):
        model = vt3d.Tokenizer3D(
            16,
            8,
            4,
            init_token=False,
            head=4,
            norm_layer_1d=vt3d.get_norm("BN", dim=1),
            norm_layer_3d=vt3d.get_norm("BN", dim=3),
        ).eval()
        model = fuse_utils.fuse_model(model, inplace=False)

        feature = torch.rand([2, 4, 8, 7, 7])
        feature1 = torch.rand([2, 4, 8, 10, 7])
        token = torch.rand([2, 16, 8])
        data = (feature, token)
        data1 = (feature1, token)
        data_list = [data, data1]

        _test_fp32_scriptable(self, model, data_list)
        _test_quant_scriptable(self, model, data_list)

    def test_vt_projector3d_scriptable(self):
        model = vt3d.Projector3D(16, 4, head=4).eval()
        model = fuse_utils.fuse_model(model, inplace=False)

        feature = torch.rand([2, 4, 8, 7, 7])
        feature1 = torch.rand([2, 4, 8, 10, 7])
        token = torch.rand([2, 16, 8])
        data_list = [(feature, token), (feature1, token)]

        _test_fp32_scriptable(self, model, data_list)
        _test_quant_scriptable(self, model, data_list)

    def test_vt_transformer_scriptable(self):
        model = vt3d.Transformer(16, head=4, trans_depth=2).eval()
        model = fuse_utils.fuse_model(model, inplace=False)

        token = torch.rand([2, 16, 8])
        data_list = [(token,), (token,)]

        _test_fp32_scriptable(self, model, data_list)
        _test_quant_scriptable(self, model, data_list)


def _run_and_compare_shape(self, model1, model2, data):
    out1 = model1(*data)
    out2 = model2(*data)
    self.assertEqual(out1.shape, out2.shape)


def _test_fp32_scriptable(
    self, pytorch_model, data_list, use_trace=False, traceable_model=None
):
    traceable_model = traceable_model or pytorch_model
    if not use_trace:
        script_model = torch.jit.script(traceable_model)
    else:
        script_model = torch.jit.trace(
            traceable_model, data_list[0], check_inputs=data_list
        )
    print(script_model.code)

    for data in data_list:
        _run_and_compare_shape(self, pytorch_model, script_model, data)


def _test_quant_scriptable(
    self, pytorch_model, data_list, use_trace=False, traceable_model=None
):
    traceable_model = traceable_model or pytorch_model
    with torch.no_grad():
        pq = qu.PostQuantizationGraph(traceable_model)
        pq.set_quant_backend("default")
        pq.set_calibrate(data_list, len(data_list))
        if not use_trace:
            pq.script()
        else:
            pq.trace(data_list[0], check_inputs=data_list)
        quant_model = pq.convert_model()

    print(quant_model.code)

    for data in data_list:
        _run_and_compare_shape(self, pytorch_model, quant_model, data)
