#!/usr/bin/env python
"""Real training-patch CUDA/AMP/continuation test, never a paper result."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from fdmrnet.config import load_config, sha256_file
from fdmrnet.data import BraTSFixedPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.amp_step import complete_amp_optimizer_step
from fdmrnet.engine.budget import TrainingBudgetLedger
from fdmrnet.engine.checkpoint import load_checkpoint, save_checkpoint
from fdmrnet.engine.trainer import _geometry_augment
from fdmrnet.losses import CompositeFDMRNetLoss, masked_l1
from fdmrnet.models import build_model
from fdmrnet.utils.io import atomic_json_dump
from fdmrnet.utils.reproducibility import seed_everything


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def difference(a, b):
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
    if torch.is_tensor(a):
        assert a.shape == b.shape
        return float((a.double() - b.cpu().double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max((difference(x, y) for x, y in zip(a, b)), default=0.)
    assert a == b, (a, b)
    return 0.


def run(config, out):
    cfg = load_config(config)
    assert torch.cuda.is_available(), 'CUDA is required; CPU fallback is forbidden'
    assert tuple(cfg['data']['patch_size_dhw']) == (48, 96, 96), 'Unexpected frozen training patch'
    seed_everything(int(cfg['experiment']['seed']), True)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    out.mkdir(parents=True, exist_ok=False)
    manifests = {k: sha256_file(cfg['data'][f'{k}_manifest']) for k in ('train', 'val')}
    degrader = ThroughPlaneDegrader(**{k: v for k, v in cfg['degradation'].items() if k != 'axis'})
    dataset = BraTSFixedPatchDataset(cfg['data']['train_manifest'], cfg['data']['modality'], degrader,
                                   tuple(cfg['data']['patch_size_dhw']), cfg['data'].get('clip_z', 5.))
    sample = dataset[0]
    batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in sample.items()}
    model = build_model(cfg['model']).to(device).train()
    criterion = CompositeFDMRNetLoss(**cfg['loss']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                 weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    scaler = torch.amp.GradScaler('cuda')
    diagnostic_cfg = copy.deepcopy(cfg)
    diagnostic_cfg['experiment']['evidence_class'] = 'diagnostic'
    ledger = TrainingBudgetLedger(out, diagnostic_cfg, manifests)
    events = []
    torch.cuda.reset_peak_memory_stats()

    def step(inject=False):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        # Exercise the active augmentation branch at its declared displacement,
        # without changing or claiming to train the source configuration.
        geometry = {**cfg['training'].get('geometry_augmentation', {}), 'probability': 1.}
        lr, field = _geometry_augment(batch['lr'], batch['hr'].shape[-3:], geometry)
        active = {**batch, 'lr': lr, 'field_target': field}
        with torch.autocast('cuda', enabled=True):
            outputs = model(lr, tuple(batch['hr'].shape[-3:]))
            if 'pred_low' in outputs:
                loss, components = criterion(outputs, active)
            else:
                loss = masked_l1(outputs['pred'], batch['hr'], batch['brain_mask'])
                components = {'total': loss}
        assert outputs['pred'].shape == batch['hr'].shape
        assert all(torch.isfinite(v).all().item() for v in components.values())
        scaler.scale(loss).backward()
        if inject:
            next(p for p in model.parameters() if p.grad is not None).grad.view(-1)[0] = torch.inf
        result = complete_amp_optimizer_step(scaler=scaler, optimizer=optimizer, scheduler=None,
                                             parameters=model.named_parameters(), gradient_clip=cfg['training']['gradient_clip'])
        torch.cuda.synchronize()
        event = {k: result[k] for k in ('overflow', 'optimizer_updated', 'old_scale', 'new_scale')}
        event.update(loss=float(loss.detach()), seconds=time.perf_counter()-t0,
                     injected_overflow=inject, components={k: float(v.detach()) for k, v in components.items()})
        events.append(event)
        return event, result

    # Native GradScaler may need initial scale backoff. Record every attempt.
    for index in range(1, 9):
        event, result = step()
        ledger.micro_batch(1, index, 1, event['loss'], result)
        if event['optimizer_updated']:
            break
    else:
        raise RuntimeError('No finite AMP optimizer update after eight native-scale attempts')
    before = cpu_tree(model.state_dict())
    opt_before = cpu_tree(optimizer.state_dict())
    sched_before = copy.deepcopy(scheduler.state_dict())
    event, result = step(True)
    ledger.micro_batch(1, index+1, 1, event['loss'], result)
    assert event['overflow'] and not event['optimizer_updated']
    assert difference(before, model.state_dict()) == 0.
    assert difference(opt_before, optimizer.state_dict()) == 0.
    assert scheduler.state_dict() == sched_before
    del before, opt_before
    scheduler.step()
    ledger.epoch_completed(1)
    saved_states = (cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()),
                    copy.deepcopy(scheduler.state_dict()), copy.deepcopy(scaler.state_dict()))
    ck = out / 'epoch_001.pt'
    save_checkpoint(ck, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=1, best_metric=0., config=cfg, manifest_hashes=manifests, training_state=ledger.state())
    ledger.bind_checkpoint(ck)
    expected_rng = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, device=device).cpu())
    event, _ = step()
    expected = (cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()), copy.deepcopy(scaler.state_dict()))
    payload = load_checkpoint(ck, model, optimizer, scheduler, scaler, restore_rng=True, map_location=device)
    restored_diffs = dict(model=difference(saved_states[0], model.state_dict()),
                          optimizer=difference(saved_states[1], optimizer.state_dict()),
                          scheduler=difference(saved_states[2], scheduler.state_dict()),
                          scaler=difference(saved_states[3], scaler.state_dict()))
    assert max(restored_diffs.values()) == 0., restored_diffs
    assert payload['config_sha256'] == cfg['_config_sha256'] and payload['manifest_hashes'] == manifests
    restored_rng = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, device=device).cpu())
    assert difference(expected_rng, restored_rng) == 0.
    resumed = TrainingBudgetLedger(out, diagnostic_cfg, manifests, restored=payload['training_state'])
    replay, result = step()
    assert replay['optimizer_updated'], 'Continuation after overflow must successfully update'
    diffs = dict(model=difference(expected[0], model.state_dict()),
                 optimizer=difference(expected[1], optimizer.state_dict()),
                 scaler=difference(expected[2], scaler.state_dict()), loss=abs(event['loss']-replay['loss']))
    # Declared before execution, no retrospective tolerance relaxation.
    # Keep the strict trajectory threshold and its outcome. CUDA grid-sample and
    # pooling backward are warn-only nondeterministic in the source runtime;
    # exact checkpoint restoration is tested separately, never inferred by
    # loosening the trajectory threshold.
    strict_trajectory_passed = max(diffs.values()) <= 1e-6
    resumed.micro_batch(2, 1, 1, replay['loss'], result)
    scheduler.step()
    resumed.epoch_completed(2)
    final = out / 'epoch_002.pt'
    save_checkpoint(final, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=2, best_metric=0., config=cfg, manifest_hashes=manifests, training_state=resumed.state())
    budget = resumed.bind_checkpoint(final)
    report = dict(status='passed', evidence_class='diagnostic', method=cfg['model']['name'],
                  config_sha256=cfg['_config_sha256'], subject=sample['subject'],
                  hr_shape=list(batch['hr'].shape), lr_shape=list(batch['lr'].shape),
                  parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  active_geometry_branch_exercised=True, events=events, resume_max_absolute_differences=diffs,
                  resume_tolerance=1e-6, strict_trajectory_passed=strict_trajectory_passed,
                  exact_checkpoint_state_restore_differences=restored_diffs,
                  pass_scope='Exact model/optimizer/scheduler/scaler/RNG restoration and successful resumed update; strict trajectory outcome reported separately',
                  rng_restored=True, forced_overflow_preserved_model_and_optimizer=True,
                  executed_budget=budget, final_checkpoint=str(final), final_checkpoint_sha256=sha256_file(final),
                  MRI_formal_training_started=False)
    atomic_json_dump(report, out / 'report.json')
    print(json.dumps({k: report[k] for k in ('status','method','peak_allocated_mib','resume_max_absolute_differences')}), flush=True)
    del model, optimizer, expected, batch, saved_states
    gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    try:
        for config in args.config:
            method = load_config(config)['model']['name']
            reports.append(run(config, out / method))
        atomic_json_dump(dict(status='passed', reports=reports), out / 'report.json')
    except Exception as exc:
        atomic_json_dump(dict(status='failed', error=repr(exc), traceback=traceback.format_exc(), reports=reports), out / 'report.json')
        raise


if __name__ == '__main__':
    main()
