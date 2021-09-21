#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from unittest import mock

import caffe2.python._import_c_extension as C
from classy_vision.fb.dataset.classy_on_box_uru_dataset import OnBoxUruDataset  # noqa
from mobile_cv.model_zoo.datasets import dataset_factory


def get_test_task_config():
    hashtag_file = "configs/cvpr_17k_equal_weight.json" # noqa
    return {
        "name": "classy_uru",
        "num_epochs": 1.5,
        "test_phase_period": 2,
        "loss": {"name": "soft_target_cross_entropy"},
        "dataset": {
            "train": {
                "name": "on_box_uru",
                "split": "train",
                "batchsize_per_replica": 2,
                "num_samples": 96,
                "num_threads": 1,
                "phases_per_epoch": 2,
                "use_shuffle": True,
                "transforms": [
                    {
                        "name": "image_decode_index",
                        "config": {
                            "color_space": "RGB",
                            "num_classes": 2,
                            "one_hot": True,
                            "single_label": False,
                        },
                        "transforms": [
                            {"name": "RandomResizedCrop", "size": 224},
                            {"name": "RandomHorizontalFlip"},
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                    }
                ],
                "hive_config": {
                    "use_mock": True,
                    "everstore_column": "everstore_id",
                    "label_column": "hashtags",
                    "partition_column": "partition_no",
                    "partition_column_values": {"start_offset": 0, "num_partitions": 2},
                    "download_folder": "/tmp",
                },
            },
            "test": {
                "name": "on_box_uru",
                "split": "val",
                "batchsize_per_replica": 2,
                "num_samples": 32,
                "num_threads": 1,
                "use_shuffle": False,
                "transforms": [
                    {
                        "name": "image_decode_index",
                        "config": {
                            "color_space": "RGB",
                            "num_classes": 2,
                            "one_hot": True,
                            "single_label": False,
                        },
                        "transforms": [
                            {"name": "Resize", "size": 256},
                            {"name": "CenterCrop", "size": 224},
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                    }
                ],
                "hive_config": {
                    "use_mock": True,
                    "everstore_column": "everstore_id",
                    "label_column": "hashtags",
                    "partition_column": "partition_no",
                    "partition_column_values": {"start_offset": 2, "num_partitions": 2},
                    "download_folder": "/tmp",
                },
            },
        },
        "model": {
            "name": "resnext",
            "num_blocks": [3],
            "base_width_and_cardinality": [4, 32],
            "small_input": False,
            "zero_init_bn_residuals": True,
            "heads": [
                {
                    "name": "fully_connected",
                    "unique_id": "default_head",
                    "num_classes": 2,
                    "fork_block": "block0-1",
                    "in_plane": 256,
                }
            ],
        },
        "meters": {"precision_at_k": {"topk": [1]}},
        "optimizer": {
            "name": "sgd",
            "num_epochs": 10,
            "lr": 0.1,
            "weight_decay": 1e-4,
            "momentum": 0.9,
        },
    }


@unittest.skipIf(
    C.is_asan, "Skip ASAN since torch + multiprocessing + fbcode doesn't work with asan"
)
class TestDatasetsClassy(unittest.TestCase):
    # adopt from fbcode/deeplearning/projects/classy_vision/fb/test/dataset_classy_on_box_uru_dataset_test.py
    @mock.patch("classy_vision.dataset.classy_dataset.get_world_size")
    def test_datasets_classy_uru(self, mock_get_world_size):
        mock_get_world_size.return_value = 2
        config = get_test_task_config()

        dataloader = dataset_factory.get(
            "classy", dataset_config=config["dataset"]["train"], num_workers=2
        )
        dl_iter = iter(dataloader)
        nbatches = 0
        for _ in dl_iter:
            nbatches += 1
        print("nbatches = {}".format(nbatches))
        # In train phase, num_samples is 96, batchsize_per_replica is 2,
        # phases_per_epoch is 2. The world size is 2. So, we expect 12 batches
        self.assertEqual(nbatches, 12)

        # setup iterator for test data
        dataloader = dataset_factory.get(
            "classy", dataset_config=config["dataset"]["test"], num_workers=2
        )
        dl_iter = iter(dataloader)
        nbatches = 0
        for _ in dl_iter:
            nbatches += 1
        print("nbatches = {}".format(nbatches))
        # In test phase, num_samples is 32, batchsize_per_replica is 2,
        # phases_per_epoch is 1 (default value). The world size is 2. So, we
        # expect 8 batches
        self.assertEqual(nbatches, 8)


if __name__ == "__main__":
    unittest.main()
