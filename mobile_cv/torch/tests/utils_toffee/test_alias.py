#!/usr/bin/env python3

import unittest

import torch
from caffe2.python import workspace
from mobile_cv.torch.utils_caffe2 import ws_utils
from mobile_cv.torch.utils_toffee import alias


class PlusTwoNet(torch.nn.Module):
    def forward(self, x):
        x = alias.alias(x, "input")
        y = x + 1
        y = alias.alias(y, "intermediate")
        z = y + 1
        z = alias.alias(z, "output")
        return z


class UpwardAliasPlusTwoNet(torch.nn.Module):
    def forward(self, x):
        x = alias.alias(x, "input")
        y = x + 1
        z = y + 1
        # NOTE: have to use upward alias because y has already been used to
        # produce z.
        y = alias.alias(y, "intermediate", is_backward=True)
        z = alias.alias(z, "output")
        return y, z


class TestToffeeAlias(unittest.TestCase):
    device = "cuda" if workspace.has_cuda_support else "cpu"

    def test_alias_with_name_op(self):
        plus_two_net = PlusTwoNet()

        test_vector = [
            torch.Tensor([42]),
            torch.Tensor(1, 2, 3),
            torch.Tensor(5, 2, 3),
            torch.Tensor(2, 2).to(torch.int64),
        ]

        for x in test_vector:
            x = x.to(self.device)
            y = plus_two_net(x)
            torch.testing.assert_allclose(x + 2, y)

    def _test_conversion(self, test_upward_alias):
        import io

        import onnx

        model = PlusTwoNet() if test_upward_alias else UpwardAliasPlusTwoNet()
        inputs = torch.Tensor([42, 6])
        with io.BytesIO() as f:
            torch.onnx.export(model, inputs, f, verbose=True)
            onnx_model = onnx.load_from_string(f.getvalue())

        from caffe2.python.onnx.backend import Caffe2Backend as c2

        init_net, predict_net = c2.onnx_graph_to_caffe2_net(onnx_model)

        alias.fuse_alias_placeholder(predict_net, init_net)

        with ws_utils.ScopedWS("__ws_tmp__", is_reset=True) as ws:
            input = torch.Tensor([6])
            ws.RunNetOnce(init_net)
            ws.FeedBlob("input", input)
            ws.RunNetOnce(predict_net)
            output = ws.FetchBlob("output")
            intermediate = ws.FetchBlob("intermediate")
            torch.testing.assert_allclose(output, input + 2)
            torch.testing.assert_allclose(intermediate, input + 1)

    def test_conversion(self):
        return self._test_conversion(test_upward_alias=False)

    def test_conversion_upward_alias(self):
        return self._test_conversion(test_upward_alias=True)
