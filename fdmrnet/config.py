from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    base_ref = raw.pop("base", None)
    if base_ref:
        base_cfg = load_config((path.parent / base_ref).resolve())
        base_cfg = {k: v for k, v in base_cfg.items() if not k.startswith("_")}
        cfg = _deep_merge(base_cfg, raw)
    else:
        cfg = raw
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path.resolve())
    cfg["_source_file_sha256"] = sha256_file(path)
    cfg["_config_sha256"] = canonical_hash({k: v for k, v in cfg.items() if not k.startswith("_")})
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["experiment", "data", "degradation", "model", "loss", "training", "evaluation"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    scale = int(cfg["degradation"]["scale"])
    if scale not in (2, 4):
        raise ValueError("Formal revision protocol only supports scales 2 and 4")
    if cfg["degradation"].get("axis", "depth") != "depth":
        raise ValueError("Formal protocol degrades the through-plane depth axis only")
    patch = tuple(int(x) for x in cfg["data"]["patch_size_dhw"])
    if len(patch) != 3 or patch[0] % scale:
        raise ValueError("patch_size_dhw must contain three values and depth must divide by scale")
    if cfg["data"]["modality"].lower() not in {"t1", "t2"}:
        raise ValueError("T1w and T2w are trained independently; modality must be t1 or t2")
    if cfg["data"].get("external_eval_without_seg", False) and not cfg["data"].get("external_eval_manifest"):
        raise ValueError("external_eval_without_seg requires external_eval_manifest")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def save_resolved_config(cfg: dict[str, Any], path: str | Path) -> None:
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
