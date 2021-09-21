#!/usr/bin/env python3

import argparse
import logging
import os

import numpy as np
import torch
import torchvision.models as models
from fblearner.flow.core.types_lib.visualizationmetrics import VISUALIZATION_METRICS
from fblearner.flow.util.visualization_utils import summary_writer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="test")

    parser.add_argument(
        "--log_dir",
        default="",
        type=str,
        help="log dir",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Writer will output to ./runs/ directory by default
    print("Tensorboard writter start initialization")
    writer = summary_writer(log_dir=args.log_dir)

    print("Building ResNet18")
    resnet18 = models.resnet18(False)

    print("Building dummy input")
    dummy_input = torch.autograd.Variable(torch.rand(1, 3, 224, 224))

    print("Adding graph")
    writer.add_graph(resnet18, (dummy_input,))

    # write dummy data
    print("Tensorboard writter start recording")
    for n_iter in range(1000):
        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        # data grouping by `slash`
        writer.add_scalar("data/scalar1a", dummy_s1[0], n_iter)
        writer.add_scalar("data/scalar2b", dummy_s2[0], n_iter)

        if n_iter % 100 == 0:
            logger.info(
                "[{}/10]: scalar1a: {}, scalar1b: {}".format(
                    int(n_iter / 100), dummy_s1[0], dummy_s2[0]
                )
            )
            print(
                "[{}/10]: scalar1a: {}, scalar1b: {}".format(
                    int(n_iter / 100), dummy_s1[0], dummy_s2[0]
                )
            )

        writer.add_scalars(
            "data/scalar_group",
            {
                "xsinx": n_iter * np.sin(n_iter),
                "xcosx": n_iter * np.cos(n_iter),
                "arctanx": np.arctan(n_iter),
            },
            n_iter,
        )
    print("Tensorboard finish writing")
    writer.close()
    return VISUALIZATION_METRICS.new(writer.log_dir)


if __name__ == "__main__":
    main()
