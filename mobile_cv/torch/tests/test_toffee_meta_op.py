#!/usr/bin/env python3

import unittest

import torch
from mobile_cv.torch.utils_toffee import meta_ops


class TestToffeeMetaOp(unittest.TestCase):
    def test_meta_conv(self):
        """Check that the meta conv operator runs"""
        model = meta_ops.MetaConv()
        x = torch.Tensor([0])
        model(x)

    def test_meta_conv_parameters_saved(self):
        """Check that the meta conv parameters are in the caffe2 net"""
        import io

        import onnx
        from caffe2.python.onnx.backend import Caffe2Backend as c2

        model = meta_ops.MetaConv(
            weight_cl=(-1.0, 1.0), bias_cl=(-2.0, 2.0), output_cl=(-5.0, 10.0)
        )
        input_data = torch.Tensor([0])
        with io.BytesIO() as f:
            torch.onnx.export(model, input_data, f, verbose=True)
            onnx_model = onnx.load_from_string(f.getvalue())
        init_net, predict_net = c2.onnx_graph_to_caffe2_net(onnx_model)
        assert predict_net.op[0].arg[0].name == "weight_l"
        assert predict_net.op[0].arg[0].f == -1.0
        assert predict_net.op[0].arg[1].name == "weight_h"
        assert predict_net.op[0].arg[1].f == 1.0
        assert predict_net.op[0].arg[2].name == "bias_l"
        assert predict_net.op[0].arg[2].f == -2.0
        assert predict_net.op[0].arg[3].name == "bias_h"
        assert predict_net.op[0].arg[3].f == 2.0
        assert predict_net.op[0].arg[4].name == "output_l"
        assert predict_net.op[0].arg[4].f == -5.0
        assert predict_net.op[0].arg[5].name == "output_h"
        assert predict_net.op[0].arg[5].f == 10.0

    def test_meta_output(self):
        """Check that the meta output operator runs"""
        model = meta_ops.MetaOutput()
        x = torch.Tensor([0])
        model(x)

    def test_meta_output_parameters_saved(self):
        """Check that the meta output parameters are in the caffe2 net"""
        import io

        import onnx
        from caffe2.python.onnx.backend import Caffe2Backend as c2

        model = meta_ops.MetaOutput(output_cl=(-5.0, 10.0))
        input_data = torch.Tensor([0])
        with io.BytesIO() as f:
            torch.onnx.export(model, input_data, f, verbose=True)
            onnx_model = onnx.load_from_string(f.getvalue())
        init_net, predict_net = c2.onnx_graph_to_caffe2_net(onnx_model)
        assert predict_net.op[0].arg[0].name == "output_l"
        assert predict_net.op[0].arg[0].f == -5.0
        assert predict_net.op[0].arg[1].name == "output_h"
        assert predict_net.op[0].arg[1].f == 10.0
