#!/usr/bin/env python
from __future__ import annotations

import argparse

from fdmrnet.config import load_config
from fdmrnet.evaluation import evaluate_full_volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test", "external"], required=True)
    parser.add_argument("--tag")
    parser.add_argument(
        "--save-subject",
        action="append",
        default=[],
        help="Subject ID to save as a compressed qualitative NPZ; may be repeated.",
    )
    args = parser.parse_args()
    print(
        evaluate_full_volume(
            load_config(args.config), args.checkpoint, args.split, args.tag, set(args.save_subject)
        )
    )


if __name__ == "__main__":
    main()
