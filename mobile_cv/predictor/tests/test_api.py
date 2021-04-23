#!/usr/bin/env python3

import json
import logging
import os
import unittest

import torch
import torch.nn as nn
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.predictor.api import FuncInfo, ModelInfo, PredictorInfo, create_predictor


logger = logging.getLogger(__name__)


class TestPreprocess:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return self.weight * x


class TestModelPlus1(nn.Module):
    def forward(self, x):
        return x + 1


def _save_test_model(path):
    m = TestModelPlus1()
    ts = torch.jit.script(m)
    os.makedirs(path, exist_ok=True)
    ts.save(os.path.join(path, "model.jit"))


@unittest.skip("Pending moving base ModelExportMethod from d2go to mobile_cv")
class TestAPI(unittest.TestCase):
    def test_model_info(self):
        with make_temp_directory("test_model_info") as tmp_dir:
            _save_test_model(tmp_dir)
            model_info = ModelInfo(path=tmp_dir, type="torchscript")
            # NOTE: decide if load_model is a public API or class method of ModelInfo
            from mobile_cv.predictor.model_wrappers import load_model

            model = load_model(model_info, model_root="")
            self.assertEqual(torch.tensor(2), model(torch.tensor(1)))

    def test_func_info(self):
        test_preprocess_info = FuncInfo(
            name=f"{__name__}.TestPreprocess", params={"weight": 2}
        )
        test_preprocess = test_preprocess_info.instantiate()
        self.assertEqual(4, test_preprocess(2))

    def test_predictor_info(self):
        pinfo = PredictorInfo(
            model=ModelInfo(path="some_path", type="some_type"),
        )
        dic = pinfo.to_dict()
        another_pinfo = PredictorInfo.from_dict(dic)
        self.assertTrue(isinstance(another_pinfo.model, ModelInfo))
        self.assertEqual(another_pinfo.model.type, "some_type")

    def test_create_predictor(self):
        with make_temp_directory("test_model_info") as tmp_dir:
            # define the predictor
            model_a_path = os.path.join(tmp_dir, "model_A")
            predictor_info = PredictorInfo(
                model=ModelInfo(path=model_a_path, type="torchscript"),
                preprocess_info=FuncInfo.gen_func_info(
                    TestPreprocess, params={"weight": 2.0}
                ),
            )

            # simulating exporting to predictor
            _save_test_model(model_a_path)
            with open(os.path.join(tmp_dir, "predictor_info.json"), "w") as f:
                json.dump(predictor_info.to_dict(), f)

            predictor = create_predictor(tmp_dir)
            # y = (x * 2) + 1
            self.assertEqual(torch.tensor(5), predictor(torch.tensor(2)))

    # TODO: add test case with pre-defined models and configs
