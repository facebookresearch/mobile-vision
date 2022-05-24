#!/usr/bin/env python3

import copy
import unittest

from caffe2.python import brew, model_helper
from mobile_cv.torch.utils_caffe2 import graph_transform


def create_linear_test_case():
    model = model_helper.ModelHelper("test_model")
    conv1 = brew.conv(model, "data", "conv1", 3, 64, 3)
    brew.relu(model, conv1, "relu1")
    # (data, conv1_w, conv1_b) -ConvOp-> conv1 -ReluOp-> relu1
    predict_net = copy.deepcopy(model.Proto())
    init_net = copy.deepcopy(model.InitProto())
    return predict_net, init_net


def create_fan_out_test_case():
    model = model_helper.ModelHelper("test_model")
    conv = brew.conv(model, "data", "conv", 3, 64, 3)
    brew.relu(model, conv, "relu1")
    brew.relu(model, conv, "relu2")
    # data -> -ConvOp-> conv -ReluOp-> relu1
    #                        -ReluOp-> relu2
    predict_net = copy.deepcopy(model.Proto())
    init_net = copy.deepcopy(model.InitProto())
    return predict_net, init_net


class TestRenameOpInput(unittest.TestCase):
    def test_linear(self):
        predict_net, init_net = create_linear_test_case()
        # (data, conv1_w, conv1_b) -ConvOp-> conv1 -ReluOp-> relu1

        # rename conv1 to conv_final
        graph_transform.rename_op_input(predict_net, init_net, 1, 0, "conv_final")
        self.assertEqual(predict_net.op[0].output[0], "conv_final")
        self.assertEqual(predict_net.op[1].input[0], "conv_final")

        # rename data to data_in, this will change external input
        graph_transform.rename_op_input(predict_net, init_net, 0, 0, "data_in")
        self.assertEqual(predict_net.op[0].input[0], "data_in")
        self.assertTrue("data" not in predict_net.external_input)
        self.assertTrue("data_in" in predict_net.external_input)

        # rename weight conv_w, this will change init_net
        self.assertEqual(predict_net.op[0].input[1], "conv1_w")
        self.assertTrue("conv1_w" in predict_net.external_input)
        self.assertEqual(init_net.op[0].output[0], "conv1_w")
        graph_transform.rename_op_input(predict_net, init_net, 0, 1, "conv1_w_new")
        self.assertEqual(predict_net.op[0].input[1], "conv1_w_new")
        self.assertTrue("conv1_w" not in predict_net.external_input)
        self.assertTrue("conv1_w_new" in predict_net.external_input)
        self.assertEqual(init_net.op[0].output[0], "conv1_w_new")

    def test_fan_out(self):
        predict_net, init_net = create_fan_out_test_case()
        # data -> -ConvOp-> conv -ReluOp-> relu1
        #                        -ReluOp-> relu2

        # rename conv to conv_new, should not allow renaming input of first
        # Relu, because it's used by the second one
        def illegal_call():
            graph_transform.rename_op_input(predict_net, init_net, 1, 0, "conv_new")

        self.assertRaises(graph_transform.IllegalGraphTransformError, illegal_call)


class TestRenameOpOutput(unittest.TestCase):
    def test_linear(self):
        predict_net, init_net = create_linear_test_case()
        # (data, conv1_w, conv1_b) -ConvOp-> conv1 -ReluOp-> relu1

        # rename conv1 to conv_final
        graph_transform.rename_op_output(predict_net, 0, 0, "conv_final")
        self.assertEqual(predict_net.op[0].output[0], "conv_final")
        self.assertEqual(predict_net.op[1].input[0], "conv_final")

        # rename relu1 to relu_final
        if "relu1" not in predict_net.external_output:
            predict_net.external_output.append("relu1")
        graph_transform.rename_op_output(predict_net, 1, 0, "relu_final")
        self.assertEqual(predict_net.op[1].output[0], "relu_final")
        self.assertTrue("relu_final" in predict_net.external_output)

    def test_fan_out(self):
        predict_net, init_net = create_fan_out_test_case()
        # data -> -ConvOp-> conv -ReluOp-> relu1
        #                        -ReluOp-> relu2

        # rename conv to conv_new, it should update both Relu's input
        graph_transform.rename_op_output(predict_net, 0, 0, "conv_new")
        self.assertEqual(predict_net.op[0].output[0], "conv_new")
        self.assertEqual(predict_net.op[1].input[0], "conv_new")
        self.assertEqual(predict_net.op[2].input[0], "conv_new")
