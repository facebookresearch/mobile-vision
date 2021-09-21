#!/usr/bin/env python3

"""
Models from classy vision
"""

import logging

import torch
from mobile_cv.common.utils_io import get_path_manager

# to register for model_zoo
from . import model_zoo_factory  # noqa

path_manager = get_path_manager()

logger = logging.getLogger("model_zoo.model_classy")


@model_zoo_factory.MODEL_ZOO_FACTORY.register("classy")
def classy_model(model_cfg, weight_file=None, reset_heads=False, **kwargs):
    from classy_vision.fb.models import build_model
    from classy_vision.generic.util import load_checkpoint, update_classy_model

    logger.info(f"Building classy vision model with config {model_cfg}")
    ret = build_model(model_cfg)
    if weight_file is not None:
        logger.info(f"Loading classy vision weight {weight_file}...")

        with path_manager.open(weight_file, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        # checkpoint = load_checkpoint(weight_file, device=torch.device("cpu"))

        state_load_success = update_classy_model(
            ret,
            checkpoint["classy_state_dict"]["base_model"],
            reset_heads=True,
        )
        assert (
            state_load_success
        ), "Update classy state from pretrained checkpoint was unsuccessful."

        # model_weights = torch.load(weight_file)
        # ret.set_classy_state(model_weights["classy_state_dict"]["base_model"])

    return ret


@model_zoo_factory.MODEL_ZOO_FACTORY.register("classy_resnext101_32x4d")
def classy_resnext101_32x4d(**kwargs):
    model_config = {
        "name": "resnext",
        "num_blocks": [3, 4, 23, 3],
        "base_width_and_cardinality": [4, 32],
        "small_input": False,
        "is_finetuning": False,
        "freeze_trunk": False,
        "heads": [
            {
                "name": "identity",
                "unique_id": "default_identity_head",
                "fork_block": "block3-2",
            }
        ],
    }
    return classy_model(model_config, **kwargs)


@model_zoo_factory.MODEL_ZOO_FACTORY.register("classy_prod_resnext3d")
def classy_prod_resnext3d(**kwargs):
    model_config = {
        "name": "prod_resnext3d",
        "frames_per_clip": 8,
        "input_planes": 3,
        "clip_crop_size": 112,
        "skip_transformation_type": "postactivated_shortcut",
        "residual_transformation_type": "postactivated_bottleneck_transformation",
        "num_blocks": [3, 4, 6, 3],
        # "input_key": "video",
        "stem_name": "resnext3d_stem",
        "stem_planes": 64,
        "stem_temporal_kernel": 3,
        "stem_spatial_kernel": 7,
        "stem_maxpool": False,
        "stage_planes": 256,
        "stage_temporal_kernel_basis": [[3], [3], [3], [3]],
        "temporal_conv_1x1": [False, False, False, False],
        "stage_temporal_stride": [1, 2, 2, 2],
        "stage_spatial_stride": [1, 2, 2, 2],
        "num_groups": 256,
        "width_per_group": 1,
        "num_classes": 11,
        "zero_init_residual_transform": True,
        "heads": [
            {
                "name": "fully_convolutional_linear",
                "unique_id": "default_head",
                "pool_size": [1, 1, 1],
                "activation_func": "softmax",
                "num_classes": 11,
                "fork_block": "pathway0-stage4-block2",
                "in_plane": 2048,
                "use_dropout": False,
            }
        ],
    }
    return classy_model(model_config, **kwargs)


@model_zoo_factory.MODEL_ZOO_FACTORY.register("classy_prod_resnext3d_video_audio")
def classy_prod_resnext3d_video_audio(**kwargs):
    model_config = {
        "name": "prod_resnext3d",
        "frames_per_clip": 4,
        "input_planes": 3,
        "clip_crop_size": 112,
        "skip_transformation_type": "postactivated_shortcut",
        "residual_transformation_type": "postactivated_bottleneck_transformation",
        "num_blocks": [[3, 3], [4, 4], [6, 6], [3, 3]],
        "octave_conv_ratios": [[0.5, 0], [0.5, 0], [0, 0], [0, 0]],
        "input_key": "video",
        "stem_name": "resnext3d_stem",
        "stem_planes": [64, 64],
        "stem_temporal_kernel": 3,
        "stem_spatial_kernel": 7,
        "stem_maxpool": False,
        "stage_planes": [256, 256],
        "stage_temporal_kernel_basis": [[[3], [1]], [[3], [1]], [[3], [1]], [[3], [1]]],
        "temporal_conv_1x1": [
            [False, True],
            [False, True],
            [False, True],
            [False, True],
        ],
        "stage_temporal_stride": [[1, 1], [2, 1], [2, 1], [1, 1]],
        "stage_spatial_stride": [[1, 2], [2, 2], [2, 2], [2, 2]],
        "glore_insertion_indices": [[], [], [], [2]],
        "glore_normalize": True,
        "glore_skip_gcn": False,
        "glore_zero_init_residual_transform": False,
        "num_groups": [24, 1],
        "width_per_group": [2, 64],
        "num_classes": 11,
        "zero_init_residual_transform": True,
        "heads": [
            {
                "name": "fully_convolutional_linear",
                "unique_id": "default_head",
                "pool_size": [1, 1, 1],
                "activation_func": "softmax",
                "num_classes": 11,
                "fork_block": "pathway_fusion",
                "in_plane": 256,
                "use_dropout": False,
            },
            {
                "name": "fully_convolutional_linear",
                "unique_id": "visual_head",
                "pool_size": [1, 7, 7],
                "activation_func": "softmax",
                "num_classes": 11,
                "fork_block": "glore_unit_pathway0-stage4-block2",
                "in_plane": 2048,
                "use_dropout": False,
            },
            {
                "name": "fully_convolutional_linear",
                "unique_id": "audio_head",
                "pool_size": [1, 4, 2],
                "activation_func": "softmax",
                "num_classes": 11,
                "fork_block": "pathway1-stage4-block2",
                "in_plane": 2048,
                "use_dropout": False,
            },
        ],
        "audio_input_key": "audio",
        "audio_stem": "audio_stem",
        "fusion_block_name": "av_concat",
        "fusion_output_dim": 256,
        "visual_pool_kernel": [1, 7, 7],
    }
    return classy_model(model_config, **kwargs)


@model_zoo_factory.MODEL_ZOO_FACTORY.register("classy_fbnet3d_video_audio")
def classy_prod_resnext3d_video_audio(
    arch_name, num_classes=11, head_in_dims=None, **kwargs
):
    if head_in_dims is None:
        head_in_dims = {
            "default_head": 256,
            "visual_head": 1408,
            "audio_head": 1408,
        }
    model_config = {
        "name": "fbnet_v2_multimodal",
        "arch": arch_name,
        "init_channels": {"video": 3, "audio": 1},
        "input_keys": {"video": "video", "audio": "audio"},
        "heads": [
            {
                "name": "fully_convolutional_linear",
                "unique_id": "default_head",
                "pool_size": [1, 1, 1],
                "activation_func": "softmax",
                "num_classes": num_classes,
                "fork_block": "fusion",
                "in_plane": head_in_dims["default_head"],
                "use_dropout": False,
            },
            {
                "name": "fully_convolutional_linear",
                "unique_id": "visual_head",
                "activation_func": "softmax",
                "num_classes": num_classes,
                "fork_block": "video",
                "in_plane": head_in_dims["visual_head"],
                "use_dropout": False,
            },
            {
                "name": "fully_convolutional_linear",
                "unique_id": "audio_head",
                "activation_func": "softmax",
                "num_classes": num_classes,
                "fork_block": "audio",
                "in_plane": head_in_dims["audio_head"],
                "use_dropout": False,
            },
        ],
    }
    return classy_model(model_config, **kwargs)
