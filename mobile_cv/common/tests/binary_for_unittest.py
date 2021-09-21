#!/usr/bin/env python3

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="test")

    parser.add_argument("--name", default=None, type=str, help="name")
    parser.add_argument("--machine_rank", default=None, type=str, help="machine_rank")
    parser.add_argument("--num_machines", default=None, type=str, help="num_machines")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(
        f"name {args.name}, "
        f"machine_rank {args.machine_rank}, num machines {args.num_machines}"
    )
    print("succeed")


if __name__ == "__main__":
    main()
