#!/usr/bin/env python
"""Evaluate archived weights using a new immutable protocol, never rewriting artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdmrnet.evaluation.evidence_protocol import build_file_inventory, evaluate_evidence_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Freeze current file SHA256 and geometry; no model inference")
    inventory.add_argument("--manifest", required=True)
    inventory.add_argument("--path-map", required=True)
    inventory.add_argument("--modality", required=True, choices=["t1", "t2", "both"])
    inventory.add_argument("--subject", action="append")
    inventory.add_argument("--out", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--out-dir", required=True)
    evaluate.add_argument("--tag", required=True)
    evaluate.add_argument("--device", default="cpu", help="cpu, cuda:0, or another explicit CUDA index")
    evaluate.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.command == "inventory":
        path = Path(args.out)
        if path.exists():
            raise RuntimeError("Refusing to overwrite an existing image inventory")
        payload = build_file_inventory(args.manifest, args.path_map, args.modality, args.subject)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(path)
    else:
        print(evaluate_evidence_protocol(args.protocol, args.out_dir, args.tag, device=args.device, threads=args.threads))


if __name__ == "__main__":
    main()
