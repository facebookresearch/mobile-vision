#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import unittest

import mobile_cv.arch.utils.fx_utils as fu
import torch
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx


class SubModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return torch.nn.functional.relu(self.conv1(x))


class SubModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(torch.nn.functional.relu(self.bn1(self.conv1(x))))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub1 = SubModel1()
        self.sub2 = SubModel2()

    def forward(self, x):
        return self.sub2(self.sub1(x))


def quantize(model, inputs, qconfig_dict):
    model = prepare_fx(model, qconfig_dict)
    model(inputs)
    qmodel = convert_fx(model)
    return qmodel


class TestUtilsFXUtils(unittest.TestCase):
    def test_get_submodule_nodes(self):
        model = Model().eval()

        gm = torch.fx.symbolic_trace(model)
        gm.graph.print_tabular()

        nodes = fu.get_submodule_nodes(gm, "sub1", ["x"], ["relu"])
        self.assertEqual(len(nodes), 2)
        self.assertEqual({x.name for x in nodes}, {"sub1_conv1", "relu"})

        nodes = fu.get_submodule_nodes(gm, "sub2", ["relu"], ["sub2_conv2"])
        self.assertEqual(len(nodes), 4)
        self.assertEqual(
            {x.name for x in nodes}, {"sub2_conv1", "sub2_bn1", "relu_1", "sub2_conv2"}
        )

    def test_extract_submodule_as_model(self):
        model = Model().eval()
        data = torch.ones(1, 1, 2, 2)

        gm = torch.fx.symbolic_trace(model)
        gm.graph.print_tabular()
        print(gm.code)

        new_model = fu.extract_submodule_as_model(gm, "sub1", ["x"], ["relu"])
        new_model = fu.extract_submodule_as_model(
            new_model, "sub2", ["sub1"], ["sub2_conv2"]
        )

        print(new_model)
        print(new_model.sub1)
        print(new_model.sub2)

        gt_out = model(data)
        new_out = new_model(data)
        self.assertEqual((gt_out - new_out).norm(), 0)

    def test_extract_quantized_submodule_quant_separately(self):
        model = Model().eval()
        data = torch.ones(1, 1, 2, 2)

        qconfig = get_default_qconfig("qnnpack")
        qconfig_dict = {"": qconfig}

        # quantize the two sub modules separately
        qmodel = copy.deepcopy(model)
        qmodel.sub1 = quantize(qmodel.sub1, data, qconfig_dict)
        qmodel.sub2 = quantize(qmodel.sub2, data, qconfig_dict)
        qmodel = torch.fx.symbolic_trace(qmodel)

        print(qmodel)
        qmodel.graph.print_tabular()

        # find input/output nodes for model.sub1
        subnodes1 = fu.get_subnodes_for_quantized_module(qmodel, "sub1")
        inputs1 = fu.get_inputs_for_quantized_submodule(subnodes1)
        outputs1 = fu.get_outputs_for_quantized_submodule(subnodes1)
        self.assertEqual(len(subnodes1), 5)
        self.assertEqual({x.name for x in inputs1}, {"x"})
        self.assertEqual({x.name for x in outputs1}, {"dequantize"})

        # find input/output nodes for model.sub2
        subnodes2 = fu.get_subnodes_for_quantized_module(qmodel, "sub2")
        inputs2 = fu.get_inputs_for_quantized_submodule(subnodes2)
        outputs2 = fu.get_outputs_for_quantized_submodule(subnodes2)
        self.assertEqual(len(subnodes2), 6)
        self.assertEqual({x.name for x in inputs2}, {"dequantize"})
        self.assertEqual({x.name for x in outputs2}, {"dequantize_1"})

        # extract sub modules
        nqmodel = fu.extract_quantized_submodule_as_model(qmodel, "sub1")
        nqmodel = fu.extract_quantized_submodule_as_model(nqmodel, "sub2")

        print(nqmodel)
        print(nqmodel.sub1)
        print(nqmodel.sub2)

        gt_out = qmodel(data)
        new_out = nqmodel(data)
        self.assertEqual((gt_out - new_out).norm(), 0)

    def test_extract_quantized_submodule_quant_together(self):
        model = Model().eval()
        data = torch.ones(1, 1, 2, 2)

        # only quantize model.sub1, it will not have forward function
        qconfig = get_default_qconfig("qnnpack")
        qconfig_dict = {"module_name": [("sub1", qconfig)]}
        qmodel = quantize(model, data, qconfig_dict)

        print(qmodel)
        qmodel.graph.print_tabular()

        # find input/output nodes for model.sub1
        subnodes1 = fu.get_subnodes_for_quantized_module(qmodel, "sub1")
        inputs1 = fu.get_inputs_for_quantized_submodule(subnodes1)
        outputs1 = fu.get_outputs_for_quantized_submodule(subnodes1)
        self.assertEqual(len(subnodes1), 5)
        self.assertEqual({x.name for x in inputs1}, {"x"})
        self.assertEqual({x.name for x in outputs1}, {"dequantize"})

        # extract sub modules
        nqmodel = fu.extract_quantized_submodule_as_model(qmodel, "sub1")

        print(nqmodel)
        print(nqmodel.sub1)
        print(nqmodel.sub2)

        gt_out = qmodel(data)
        new_out = nqmodel(data)
        self.assertEqual((gt_out - new_out).norm(), 0)
