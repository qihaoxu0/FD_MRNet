#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


FORBIDDEN = (b"SIMULATED", b"PLACEHOLDER", b"NOT FOR SUBMISSION", b"NOT_FOR_SUBMISSION")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    bad = []
    for root in map(Path, args.paths):
        files = root.rglob("*") if root.is_dir() else [root]
        for path in files:
            if not path.is_file(): continue
            if any(token.decode().lower() in path.name.lower() for token in FORBIDDEN):
                bad.append(str(path)); continue
            if path.suffix.lower() in {".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}:
                data = path.read_bytes()
                if any(token in data.upper() for token in FORBIDDEN): bad.append(str(path))
    if bad:
        raise SystemExit("Submission artifact verification failed:\n" + "\n".join(sorted(set(bad))))
    print("Artifact verification passed: no placeholder markers found")


if __name__ == "__main__": main()

