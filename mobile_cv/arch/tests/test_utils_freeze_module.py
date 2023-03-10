#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import tempfile
import unittest

import torch
from mobile_cv.arch.utils.freeze_module import FreezeContainer, FreezeModule


class Model(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.module = module

    def forward(self, x):
        return self.module(self.conv(x))


class ModelCastFp32(torch.nn.Module):
    def forward(self, x):
        return x.float()


def _set_model_weights(model: torch.nn.Module, value: float = 0.00001):
    for x in model.parameters():
        torch.nn.init.constant_(x, value)


class ModuleToFreeze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self, x):
        return self.bn(self.conv(x))


class TestUtilsFreezeModule(unittest.TestCase):
    def test_freeze_container(self):
        fm1 = ModuleToFreeze()
        _set_model_weights(fm1, 0.05)
        model1 = Model(FreezeContainer(fm1))
        # model1 = Model(fm1)
        _set_model_weights(model1, 0.01)

        print(list(model1.named_parameters()))
        print(model1.state_dict())
        print(fm1.state_dict())

        self.assertEqual(len(list(model1.named_parameters())), 2)
        self.assertEqual(len(model1.state_dict()), 2)

        inputs = torch.randn(4, 1, 1, 1)
        model1.train()
        for _ in range(5):
            model1(inputs)

        print(list(model1.named_parameters()))
        print(model1.state_dict())
        print(fm1.state_dict())

        self.assertEqual(fm1.bn.num_batches_tracked, 0)
        self.assertEqual(fm1.bn.running_mean.detach().item(), 0)
        self.assertAlmostEqual(fm1.bn.weight.detach().item(), 0.05)

    def test_freeze_container_save_load(self):
        fm1 = ModuleToFreeze()
        _set_model_weights(fm1, 0.05)
        model1 = Model(FreezeContainer(fm1))
        _set_model_weights(model1, 0.01)

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model1.state_dict(), os.path.join(tmpdirname, "cp.pth"))

            fm2 = ModuleToFreeze()
            _set_model_weights(fm2, 0.25)
            model2 = Model(FreezeContainer(fm2))
            _set_model_weights(model2, 0.45)

            model2_state_dict = torch.load(os.path.join(tmpdirname, "cp.pth"))
            model2.load_state_dict(model2_state_dict)

            print(list(model2.named_parameters()))

            self.assertAlmostEqual(model2.conv.weight.detach().item(), 0.01)
            self.assertEqual(model2.module.module.bn.weight.detach().item(), 0.25)

    def test_freeze_container_device(self):
        if not torch.cuda.is_available():
            return

        fm1 = ModuleToFreeze().cuda()
        model1 = Model(torch.nn.Sequential(ModelCastFp32(), FreezeModule(fm1))).cuda()

        out1 = model1(torch.randn(4, 1, 1, 1).cuda())
        print(out1.dtype)

        model1.half()
        out2 = model1(torch.randn(4, 1, 1, 1).cuda().half())
        print(out2.dtype)
