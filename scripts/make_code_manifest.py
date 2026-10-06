#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", "outputs", "generated_configs", "placeholders"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="CODE_MANIFEST.sha256")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() == out:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts) or path.suffix == ".pyc":
            continue
        files.append(relative)
    lines = [f"{digest(root / relative)}  {relative.as_posix()}" for relative in sorted(files)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} hashes to {out}")


if __name__ == "__main__":
    main()
