#!/usr/bin/env python
from __future__ import annotations

import argparse

from fdmrnet.config import load_config
from fdmrnet.engine.trainer import train_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    train_from_config(load_config(args.config), args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()

