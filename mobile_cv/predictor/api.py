#!/usr/bin/env python3

import copy
import json
import logging
import os
from typing import Any, Dict, NamedTuple, Optional

import torch.nn as nn
from mobile_cv.common import utils_io
from mobile_cv.common.misc.py import dynamic_import
from mobile_cv.predictor.builtin_functions import (
    IdentityPostprocess,
    IdentityPreprocess,
    NaiveRunFunc,
)


path_manager = utils_io.get_path_manager()
logger = logging.getLogger(__name__)


class ModelInfo(NamedTuple):
    """
    Relevant information in order to load an model.

    Args:
        path (str): A single string to identify the (relative) path of model, it can be
            a single file (eg. a torchscript) or a directory (eg. caffe2 model that has
            predict and init net).
        export_method (str): The name of ModelExportMethod which is responsible to load
            the exported model.
        load_kwargs (dict): The kwargs that will be used when loading the exported model
    """

    path: str
    export_method: str
    load_kwargs: Dict[str, Any]


class FuncInfo(NamedTuple):
    """
    Relevant information to construct an callable object.

    NOTE: Currently the object has to be a class (not function).

    Args:
        name (str): the name to identify the function, for python it's the full module
            name and object name (eg. foo.bar.MyPreprocessFunc).
        params (Dict[str, Any]): the kwargs used to construct the object.
    """

    name: str
    params: Dict[str, Any]

    def instantiate(self):
        class_obj = dynamic_import(self.name)
        return class_obj(**self.params)

    @staticmethod
    def gen_func_info(class_obj, params):
        return FuncInfo(
            name="{}.{}".format(class_obj.__module__, class_obj.__qualname__),
            params=params,
        )


class PredictorInfo(NamedTuple):
    """
    Relevant information to create an Predictor. The Predictor can contain one or many
    ML models in various formats (caffe2, torchscript, boltnn, etc.) and allow
    customized preprocess/postprocess and execution logic.

    The execution of y = predictor(x) will be in the order of:
        inputs = preprocess(x)
        outputs = run_func(model_or_models, inputs)
        y = postprocess(x, inputs, outputs)
    """

    # a single model or a dict of sub models
    model: Optional[ModelInfo] = None
    models: Optional[Dict[str, ModelInfo]] = None
    preprocess_info: FuncInfo = FuncInfo.gen_func_info(IdentityPreprocess, params={})
    postprocess_info: FuncInfo = FuncInfo.gen_func_info(IdentityPostprocess, params={})
    run_func_info: FuncInfo = FuncInfo.gen_func_info(NaiveRunFunc, params={})

    @staticmethod
    def from_dict(dic):
        dic = copy.deepcopy(dic)
        # maybe there's a way to avoid explicitly cast everything
        if "model" in dic:
            dic["model"] = ModelInfo(**dic["model"])
        if "models" in dic:
            dic["models"] = {k: ModelInfo(**v) for k, v in dic["models"].items()}
        dic["preprocess_info"] = FuncInfo(**dic["preprocess_info"])
        dic["postprocess_info"] = FuncInfo(**dic["postprocess_info"])
        dic["run_func_info"] = FuncInfo(**dic["run_func_info"])
        return PredictorInfo(**dic)

    def to_dict(self):
        def _to_dict(x):
            ret = {}
            if hasattr(x, "_fields"):  # NamedTuple like ModelInfo, FuncInfo
                x = x._asdict()
            if not isinstance(x, dict):
                return x
            for k, v in x.items():
                if v is None:
                    # don't store None items in predictor_info.json
                    continue
                ret[k] = _to_dict(v)
            return ret

        return _to_dict(self)


class PredictorWrapper(nn.Module):
    def __init__(self, model_or_models, run_func, preprocess, postprocess):
        super().__init__()
        self.model_or_models = model_or_models
        self.run_func = run_func
        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):  # NOTE: support only single input and output
        inputs = self.preprocess(x)
        outputs = self.run_func(self.model_or_models, inputs)
        y = self.postprocess(x, inputs, outputs)
        return y

    def get_wrapped_models(self):
        """Return the torchscript model directly
        Note:
          - This method only support single model without customizing RunFunc
          - If using model wrapper, get_wrapped_models need to be defined
        """
        model = self.model_or_models
        assert not isinstance(model, dict), (
            "This method only support single model. "
            + "Please define multiple exporter types for multi-model"
        )
        assert isinstance(
            self.run_func, NaiveRunFunc
        ), "Customized RunFunc is not supported, please using Proprocessing & Postprocessing"
        while hasattr(model, "get_wrapped_models"):
            # allow customized wrapper, e.g. TracingAdapterModelWrapper, TorchscriptWrapper
            model = model.get_wrapped_models()
        return model


def create_predictor(predictor_dir):
    logger.info("Creating predictor from structured folder: {}".format(predictor_dir))
    return _create_predictor(
        info_json=os.path.join(predictor_dir, "predictor_info.json"),
        model_root=predictor_dir,
    )


def _create_predictor(info_json, model_root):
    logger.info("Loading predictor info from {}".format(info_json))
    with path_manager.open(info_json) as f:
        info_dict = json.load(f)
        predictor_info = PredictorInfo.from_dict(info_dict)

    def _load_from_model_info(model_info, root):
        logger.info("Loading from ModelInfo: {}".format(model_info))
        model_export_method = dynamic_import(model_info.export_method)
        # join root and model_info.path and collapse duplicated "." from path, eg:
        # "uri_prefix://root_dir/./model_dir" -> "uri_prefix://root_dir/model_dir"
        fake_full_path = os.path.join("FAKE_ROOT", model_info.path)
        rel_paths = os.path.normpath(fake_full_path).split(os.sep)[1:]
        save_path = os.path.join(root, *rel_paths)
        return model_export_method.load(save_path, **model_info.load_kwargs)

    assert (predictor_info.model is None) ^ (predictor_info.models is None)
    if predictor_info.model is not None:
        model_or_models = _load_from_model_info(predictor_info.model, model_root)
    else:
        model_or_models = {
            k: _load_from_model_info(info, model_root)
            for k, info in predictor_info.models.items()
        }

    return PredictorWrapper(
        model_or_models=model_or_models,
        run_func=predictor_info.run_func_info.instantiate(),
        preprocess=predictor_info.preprocess_info.instantiate(),
        postprocess=predictor_info.postprocess_info.instantiate(),
    )
