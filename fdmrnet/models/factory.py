from __future__ import annotations

import inspect

from .baselines import EDSR3D, InterpolationBaseline, MatchedResidual3D, RDN3D, SpectralSR3D, SwinIR3D
from .fdmrnet import FDMRNet


def build_model(config: dict):
    cfg = dict(config)
    name = cfg.pop("name").lower()
    registry = {
        "fdmrnet": FDMRNet,
        "edsr3d": EDSR3D,
        "rdn3d": RDN3D,
        "swinir3d": SwinIR3D,
        "spectralsr3d": SpectralSR3D,
        "matched_residual3d": MatchedResidual3D,
        "trilinear": InterpolationBaseline,
    }
    if name not in registry:
        raise KeyError(f"Unknown model {name}; choices={sorted(registry)}")
    constructor = registry[name]
    if name == "trilinear":
        return constructor()
    # Config inheritance intentionally keeps the complete formal protocol.  Baseline
    # constructors must not receive FD-MRNet-only keys such as max_coarse_disp.
    signature = inspect.signature(constructor.__init__)
    accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if not accepts_extra:
        allowed = {key for key in signature.parameters if key != "self"}
        cfg = {key: value for key, value in cfg.items() if key in allowed}
    return constructor(**cfg)
