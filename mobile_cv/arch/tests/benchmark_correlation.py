#!/usr/bin/env python3

import logging
import time
from typing import Tuple

import torch
import torch.nn as nn
from mobile_cv.arch.fbnet_v2.correlation import (
    MatMulCorrelationBlock,
    NaiveCorrelationBlock,
    TorchscriptNaiveCorrelation,
    TorchscriptUnfoldCorrelation,
    UnfoldCorrelationBlock,
)


class BenchmarkCorrelationBlock:
    @staticmethod
    def _benchmark_correlation(
        correlation_block: nn.Module,
        image_shape: Tuple[int, int, int, int],
        benchmark_iter_max: int,
    ) -> None:

        model = correlation_block(k=0, d=1, s1=1, s2=1)
        x1 = torch.randn(*image_shape)
        x2 = torch.randn(*image_shape)

        start = time.process_time()
        for _ in range(benchmark_iter_max):
            model(x1, x2)
        end = time.process_time()

        # pyre-fixme[7]: Expected `None` but got `float`.
        # pyre-fixme[7]: Expected `None` but got `float`.
        return (end - start) / benchmark_iter_max

    @staticmethod
    def benchmark() -> None:
        image_shape = [4, 2, 8, 8]
        logging.info(
            f"Image shape is B={image_shape[0]}, C={image_shape[1]}, H={image_shape[2]}, W={image_shape[3]}."
        )

        benchmark_iter_max = 10
        logging.info(f"Benchmarking is done over {benchmark_iter_max} iterations.")

        modules = [
            UnfoldCorrelationBlock,
            MatMulCorrelationBlock,
            NaiveCorrelationBlock,
            TorchscriptNaiveCorrelation,
            TorchscriptUnfoldCorrelation,
        ]
        for module in modules:
            bench_time = BenchmarkCorrelationBlock._benchmark_correlation(
                # pyre-fixme[6]: Expected `Module` for 1st param but got
                #  `Type[typing.Union[MatMulCorrelationBlock, NaiveCorrelationBlock]]`.
                # pyre-fixme[6]: Expected `Module` for 1st param but got
                #  `Type[typing.Union[MatMulCorrelationBlock, NaiveCorrelationBlock]]`.
                module,
                image_shape,
                benchmark_iter_max,
            )
            logging.info(f"Time taken by module {module.__name__} is {bench_time}.")


def main() -> None:
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s] %(message)s", level=logging.INFO
    )
    logging.info("Starting benchmarking for correlation module.")
    BenchmarkCorrelationBlock.benchmark()


if __name__ == "__main__":
    main()
