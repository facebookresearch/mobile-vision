#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.utils.quantize_utils as qu
import mock
import torch
from mobile_cv.arch.layers import NaiveSyncBatchNorm
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


class TestModule(qu.QuantizableModule):
    def __init__(self):
        super().__init__(eager_mode=True, qconfig=None, n_inputs=2, n_outputs=1)

    @qu.QuantizableModule.quant_dequant()
    def forward(self, x, y):
        return x + y

    @qu.QuantizableModule.dequant_quant()
    def forward_dq(self, x):
        return x / 2.0, x * 2.0


class TestUtilsQuantizeUtils(unittest.TestCase):
    def test_post_quantization(self):
        e6 = {"expansion": 6}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1)],
                # stage 1
                [("ir_k3", 8, 2, 2, e6), ("ir_k5", 8, 1, 1, e6)],
            ]
        }
        model = _build_model(arch_def, 3)
        data = torch.rand((4, 3, 8, 8))
        org_out = model(data)

        pq = qu.PostQuantization(model)
        pq.fuse_bn().add_quant_stub().set_quant_backend("fbgemm")
        quant_model = pq.prepare().calibrate_model([[data]], 1).convert_model()

        quant_out = quant_model(data)
        self.assertEqual(quant_out.shape, org_out.shape)

        org_out_after = model(data)
        # make sure the original model was not modifed
        self.assertEqual(org_out.norm(), org_out_after.norm())

    def test_utils_quantize_utils_quantize_model_graph_mode(self):
        e6 = {"expansion": 6}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1)],
                # stage 1
                [("ir_k3", 8, 2, 2, e6), ("ir_k5", 8, 1, 1, e6)],
            ]
        }
        model = _build_model(arch_def, 3)
        data = torch.rand((4, 3, 8, 8))
        org_out = model(data)

        pq = qu.PostQuantizationGraph(model)
        pq.set_quant_backend("fbgemm")
        quant_model = pq.set_calibrate([[data]], 1).trace([data]).convert_model()

        print(quant_model)

        quant_out = quant_model(data)
        self.assertEqual(quant_out.shape, org_out.shape)

        org_out_after = model(data)
        # make sure the original model was not modifed
        self.assertEqual(org_out.norm(), org_out_after.norm())

    def test_utils_quantize_utils_quantstub_nested(self):
        class AddStub(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        qsl = qu.QuantStubNested.FromCount(1, AddStub)
        qsl.eval()
        inputs = torch.ones((1,))
        output = qsl(inputs)
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output, 2.0)

        qsl = qu.QuantStubNested.FromCount(3, AddStub)
        qsl.eval()
        inputs = [torch.ones((1,)), torch.ones((1,)) * 2, torch.ones((1,)) * 3]

        # single input as a list
        output = qsl(inputs)
        self.assertTrue(isinstance(output, list))
        self.assertEqual(len(output), 3)
        self.assertEqual(output[0], 2.0)
        self.assertEqual(output[1], 3.0)
        self.assertEqual(output[2], 4.0)

        inputs1 = {
            "m1": torch.ones((1,)),
            "m2": torch.ones((1,)) * 2,
            "m3": torch.ones((1,)) * 3,
        }
        # single input as a dict
        output = qsl(inputs1)
        self.assertEqual(len(output), 3)
        self.assertEqual(output["m1"], 2.0)
        self.assertEqual(output["m2"], 3.0)
        self.assertEqual(output["m3"], 4.0)

        # two inputs with a list and a dict
        output1, output2 = qsl((inputs[:2], {"m3": inputs[-1]}))
        self.assertEqual(output1[0], 2.0)
        self.assertEqual(output1[1], 3.0)
        self.assertEqual(output2["m3"], 4.0)

    def test_utils_quantize_utils_quantizable_module(self):

        with mock.patch("mobile_cv.arch.utils.quantize_utils.QuantStub.forward") as qs:
            with mock.patch(
                "mobile_cv.arch.utils.quantize_utils.DeQuantStub.forward"
            ) as dqs:
                qs.side_effect = lambda x: x
                dqs.side_effect = lambda x: x

                model = TestModule()
                input1 = torch.ones((1,))
                input2 = torch.ones((1,))
                output = model(input1, input2)
                self.assertEqual(output, 2.0)
                self.assertEqual(qs.call_count, 2)
                self.assertEqual(dqs.call_count, 1)

                r1, r2 = model.forward_dq(output)
                self.assertEqual(r1, 1.0)
                self.assertEqual(r2, 4.0)
                self.assertEqual(qs.call_count, 4)
                self.assertEqual(dqs.call_count, 2)

    def test_wrap_quant_subclass(self):
        class AddList(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def forward(self, x):
                assert isinstance(x, list)
                return [x[0] + self.val, x[1] + self.val]

            def mul(self, x):
                assert isinstance(x, list)
                return [x[0] * self.val, x[1] * self.val]

        model = AddList(2)
        wrapped = qu.wrap_quant_subclass(model, 2, 2)
        self.assertTrue(isinstance(wrapped, AddList))
        self.assertTrue(issubclass(type(wrapped), AddList))

        inputs = [torch.ones((1,)), torch.ones((1,)) * 2.0]
        outputs = wrapped(inputs)
        self.assertTrue(isinstance(outputs, list))
        self.assertEqual(len(outputs), 2)

        outputs1 = wrapped.mul(inputs)
        self.assertTrue(isinstance(outputs1, list))
        self.assertEqual(len(outputs1), 2)

    def test_wrap_quant_subclass_multisubclass(self):
        class AddList(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def forward(self, x):
                assert isinstance(x, list)
                return [x[0] + self.val, x[1] + self.val]

            def mul(self, x):
                assert isinstance(x, list)
                return [x[0] * self.val, x[1] * self.val]

        class AddAddList(AddList):
            pass

        model = AddAddList(2)
        wrapped = qu.wrap_quant_subclass(model, 2, 2)
        self.assertTrue(isinstance(wrapped, AddAddList))
        self.assertTrue(issubclass(type(wrapped), AddAddList))

        inputs = [torch.ones((1,)), torch.ones((1,)) * 2.0]
        outputs = wrapped(inputs)
        self.assertTrue(isinstance(outputs, list))
        self.assertEqual(len(outputs), 2)

        outputs1 = wrapped.mul(inputs)
        self.assertTrue(isinstance(outputs1, list))
        self.assertEqual(len(outputs1), 2)

    def test_quant_warpper_with_kwargs(self):

        model = TestModule()
        input1 = torch.ones((1,))
        input2 = torch.ones((1,))

        # run with kwargs
        output = model(input1, y=input2)
        self.assertEqual(output, 2.0)

    def test_quant_warpper_bypass_kwargs(self):
        class Module(qu.QuantizableModule):
            def __init__(self):
                super().__init__(eager_mode=True, qconfig=None, n_inputs=2, n_outputs=1)

            @qu.QuantizableModule.quant_dequant(bypass_kwargs=["s"])
            def forward(self, x, y, s):
                return x + y + len(s)

        model = Module()
        input1 = torch.ones((1,))
        input2 = torch.ones((1,))
        input3 = "len_is_8"

        output = model(input1, y=input2, s=input3)
        self.assertEqual(output, 10.0)

    def _test_bn_swap_impl(self, bn_source_cls, conversion_fn):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = bn_source_cls(1)
                self.nested = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 1, 1),
                    bn_source_cls(1),
                )

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        m = M()

        # initialize BN params
        m.bn.weight = torch.nn.Parameter(torch.Tensor([0.123]))
        m.bn.bias = torch.nn.Parameter(torch.Tensor([0.456]))
        m.nested[0].weight = torch.nn.Parameter(torch.Tensor([0.321]))
        m.nested[0].bias = torch.nn.Parameter(torch.Tensor([0.654]))

        # initialize BN running stats
        data = torch.randn(32, 1, 32, 32)
        m(data)

        # swap
        m_copy = copy.deepcopy(m)
        conversion_fn(m_copy)

        # verify equivalency in train mode
        data2 = torch.randn(32, 1, 32, 32)
        self.assertTrue(
            torch.allclose(m(data2), m_copy(data2)),
            "numerics changed after BN swap in train mode",
        )
        # verify equivalency in eval mode
        m.eval()
        m_copy.eval()
        data3 = torch.randn(32, 1, 32, 32)
        self.assertTrue(
            torch.allclose(m(data3), m_copy(data3)),
            "numerics changed after BN swap in eval mode",
        )

    def test_syncbn_to_bn_swap(self):
        """Verifies that the NaiveSyncBatchNorm to BatchNorm2d swap is working correctly."""
        self._test_bn_swap_impl(NaiveSyncBatchNorm, qu.swap_syncbn_to_bn)

    def test_bn_to_syncbn_swap(self):
        """Verifies that the BatchNorm2d to NaiveSyncBatchNorm swap is working correctly."""
        self._test_bn_swap_impl(torch.nn.BatchNorm2d, qu.swap_bn_to_syncbn)

    def test_get_qconfig_dict(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return self.conv(self.conv1(x))

            def get_qconfig_dict(self, qconfig):
                return {"": qconfig, "module_name": [("conv", None)]}

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_m2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return self.conv_m2(x)

            def get_qconfig_dict(self, qconfig):
                return {
                    "": None,
                }

        class M3(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class MM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ma = torch.nn.Sequential(
                    M1(),
                    M2(),
                    M3(),
                )
                self.mb = M1()

            def forward(self, x):
                return self.mb(self.ma(x))

        qconfig = "test"
        qconfig_dict = qu.get_qconfig_dict(MM(), qconfig)
        gt_qconfig_dict = {
            "": qconfig,
            "module_name": [
                ("ma", qconfig),
                ("ma.0", qconfig),
                ("ma.0.conv", None),
                ("ma.1", None),
                ("mb", qconfig),
                ("mb.conv", None),
            ],
        }
        # check the qconfig_dict
        self.assertEqual(qconfig_dict, gt_qconfig_dict)

        # quantize and convert the model
        model = MM().eval()
        qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        qconfig_dict = qu.get_qconfig_dict(model, qconfig)
        model = prepare_fx(model, qconfig_dict)
        model = convert_fx(model)
        print(model)

        self.assertIsInstance(
            getattr(model.ma, "0").conv1, torch.nn.quantized.modules.conv.Conv2d
        )
        self.assertIsInstance(getattr(model.ma, "0").conv, torch.nn.Conv2d)
        self.assertIsInstance(getattr(model.ma, "1").conv_m2, torch.nn.Conv2d)
        self.assertIsInstance(model.mb.conv1, torch.nn.quantized.modules.conv.Conv2d)
        self.assertIsInstance(model.mb.conv, torch.nn.Conv2d)
