#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.utils.jit_utils as ju
import torch
import torch.nn as nn


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        sz = x[0]
        if sz % 2 == 0:
            return torch.ones(sz, sz) * (sz + 1)
        else:
            return torch.ones(sz + 1, sz + 1) * (sz + 1)


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = SubModule()

    def to_traceable(self):
        self.sub = torch.jit.script(self.sub)

    def forward(self, x):
        return self.sub(x)


class TestUtilsJitUtils(unittest.TestCase):
    def test_to_traceable(self):
        model = nn.Sequential(TestModule())
        model.eval()

        data1 = torch.tensor([3])
        data2 = torch.tensor([2])
        out1 = model(data1)
        out2 = model(data2)
        self.assertEqual(out1.shape, torch.Size([4, 4]))
        self.assertEqual(out2.shape, torch.Size([2, 2]))

        # tracing that produces incorrect results
        trace_model = torch.jit.trace(model, [data1])
        tout1 = trace_model(data1)
        tout2 = trace_model(data2)
        self.assertEqual(tout1.shape, out1.shape)
        # shape not match as the traced model could not handle control flow
        self.assertNotEqual(tout2.shape, out2.shape)
        self.assertEqual(tout2.shape, torch.Size([3, 3]))

        # tracing that produces correct results
        traceable_model = ju.get_traceable_model(model)
        trace_script_model = torch.jit.trace(traceable_model, [data1])
        tsout1 = trace_script_model(data1)
        tsout2 = trace_script_model(data2)
        self.assertEqual(tsout1.shape, out1.shape)
        self.assertEqual(tsout2.shape, out2.shape)

        # should be safe to call twice
        traceable_model1 = ju.get_traceable_model(traceable_model)
        trace_script_model1 = torch.jit.trace(traceable_model1, [data1])
        tsout11 = trace_script_model1(data1)
        tsout22 = trace_script_model1(data2)
        self.assertEqual(tsout11.shape, out1.shape)
        self.assertEqual(tsout22.shape, out2.shape)
