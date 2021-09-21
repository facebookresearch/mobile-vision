#!/usr/bin/env python3

import logging
import os
from itertools import count

import numpy as np
import torch
import torch.nn as nn
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from mobile_cv.common import utils_io
from mobile_cv.torch.utils_caffe2.protobuf import infer_device_type
from mobile_cv.torch.utils_caffe2.ws_utils import ScopedWS


path_manager = utils_io.get_path_manager()
logger = logging.getLogger(__name__)


def load_model(model_path, model_type) -> nn.Module:
    logger.info("Loading {} model from {} ...".format(model_path, model_type))

    if model_type.startswith("torchscript"):
        extra_files = {}
        # NOTE: may support loading extra_file specified by model_info
        # extra_files["predictor_info.json"] = ""

        with path_manager.open(os.path.join(model_path, "model.jit"), "rb") as f:
            ts = torch.jit.load(f, _extra_files=extra_files)

        return TorchscriptWrapper(ts)

    elif model_type.startswith("caffe2"):
        # load predict_net
        predict_net = caffe2_pb2.NetDef()
        with path_manager.open(os.path.join(model_path, "model.pb"), "rb") as f:
            predict_net.ParseFromString(f.read())

        # NOTE: may support force engine via Caffe2 specific model_info
        # if force_engine is not None:
        #     for op in predict_net.op:
        #         op.engine = force_engine

        # load init_net
        init_net = caffe2_pb2.NetDef()
        with path_manager.open(os.path.join(model_path, "model_init.pb"), "rb") as f:
            init_net.ParseFromString(f.read())

        return Caffe2Wrapper(predict_net, init_net)
    else:
        raise RuntimeError(f"Unsupported model type: {model_type}")


class Caffe2Wrapper(nn.Module):
    """
    A class works just like nn.Module in terms of inference, but running
    caffe2 model under the hood. Input/Output are Tuple[tensor].
    """

    _ids = count(0)

    def __init__(self, predict_net, init_net):
        logger.info("Initializing Caffe2Wrapper for: {} ...".format(predict_net.name))
        super().__init__()
        assert isinstance(predict_net, caffe2_pb2.NetDef)
        assert isinstance(init_net, caffe2_pb2.NetDef)
        # create unique temporary workspace for each instance of Caffe2Wrapper
        self.ws_name = "__tmp_Caffe2Wrapper_{}__".format(next(self._ids))
        self.net = core.Net(predict_net)

        logger.info("Running init_net once to fill the parameters ...")
        with ScopedWS(self.ws_name, is_reset=True, is_cleanup=False) as ws:
            ws.RunNetOnce(init_net)
            uninitialized_external_input = []
            for blob in self.net.Proto().external_input:
                if blob not in ws.Blobs():
                    uninitialized_external_input.append(blob)
                    ws.CreateBlob(blob)
            ws.CreateNet(self.net)

        self._error_msgs = set()
        self._input_blobs = uninitialized_external_input

    def _infer_output_devices(self, inputs):
        def _get_device_type(torch_tensor):
            assert torch_tensor.device.type in ["cpu", "cuda"]
            assert torch_tensor.device.index == 0
            return torch_tensor.device.type

        predict_net = self.net.Proto()
        input_device_types = {
            (name, 0): _get_device_type(tensor)
            for name, tensor in zip(self._input_blobs, inputs)
        }
        device_type_map = infer_device_type(
            predict_net, known_status=input_device_types, device_name_style="pytorch"
        )
        ssa, versions = core.get_ssa(predict_net)
        versioned_outputs = [
            (name, versions[name]) for name in predict_net.external_output
        ]
        output_devices = [device_type_map[outp] for outp in versioned_outputs]
        return output_devices

    def forward(self, inputs):
        assert len(inputs) == len(
            self._input_blobs
        ), "Number of input tensors ({}) doesn't match the required input blobs: {}".format(
            len(inputs), self._input_blobs
        )
        with ScopedWS(self.ws_name, is_reset=False, is_cleanup=False) as ws:
            # Feed inputs
            for b, tensor in zip(self._input_blobs, inputs):
                # feed torch.Tensor directly, maybe need to cast to numpy first
                ws.FeedBlob(b, tensor)
            # Run predict net
            try:
                ws.RunNet(self.net.Proto().name)
            except RuntimeError as e:
                if not str(e) in self._error_msgs:
                    self._error_msgs.add(str(e))
                    logger.warning("Encountered new RuntimeError: \n{}".format(str(e)))
                logger.warning("Catch the error and use partial results.")

            c2_outputs = [ws.FetchBlob(b) for b in self.net.Proto().external_output]
            # Remove outputs of current run, this is necessary in order to
            # prevent fetching the result from previous run if the model fails
            # in the middle.
            for b in self.net.Proto().external_output:
                # Needs to create uninitialized blob to make the net runable.
                # This is "equivalent" to: ws.RemoveBlob(b) then ws.CreateBlob(b),
                # but there'no such API.
                ws.FeedBlob(
                    b, f"{b}, a C++ native class of type nullptr (uninitialized)."
                )

        # Cast output to torch.Tensor on the desired device
        output_devices = (
            self._infer_output_devices(inputs)
            if any(t.device.type != "cpu" for t in inputs)
            else ["cpu" for _ in self.net.Proto().external_output]
        )
        outputs = []
        for name, c2_output, device in zip(
            self.net.Proto().external_output, c2_outputs, output_devices
        ):
            if not isinstance(c2_output, np.ndarray):
                raise RuntimeError(
                    "Invalid output for blob {}, received: {}".format(name, c2_output)
                )
            outputs.append(torch.Tensor(c2_output).to(device=device))

        return tuple(outputs)


class TorchscriptWrapper(nn.Module):
    """A class that works like nn.Module in terms of inference, but running
    Torchscript model under the hood. Input/Output are Tuple[tensor]."""

    def __init__(self, module, int8_backend=None):
        super().__init__()
        self.module = module
        self.int8_backend = int8_backend

    def forward(self, *args, **kwargs):
        # TODO: set int8 backend accordingly if needed
        return self.module(*args, **kwargs)
