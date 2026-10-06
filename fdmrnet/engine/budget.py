"""Executed counters for independently checkpoint-bound training evidence."""
from __future__ import annotations

import json
import importlib.metadata
import platform
import shutil
import sys
from pathlib import Path

import torch

from fdmrnet.config import canonical_hash, sha256_file
from fdmrnet.utils.io import atomic_json_dump


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'fdmrnet').rglob('*.py')) + [root / 'scripts/train.py']
    files = {str(p.relative_to(root)): sha256_file(p) for p in paths}
    return {'files': files, 'sha256': canonical_hash(files)}


class TrainingBudgetLedger:
    """One process per model; reject unimplemented distributed accounting."""
    def __init__(self, output, cfg, manifests, world_size=1, restored=None):
        if world_size != 1:
            raise RuntimeError('Executed-budget ledger currently requires one GPU/process per job')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = self.output / 'training_execution.jsonl'
        self.cfg = cfg
        self.manifests = manifests
        self.source = source_identity()
        self.runtime = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}
        self.counters = dict(micro_batches=0, optimizer_updates=0, optimizer_attempts=0,
                             skipped_optimizer_updates=0, sampled_patches=0,
                             scheduler_steps=0, epochs_completed=0)
        if restored:
            if restored['source_code_sha256'] != self.source['sha256']:
                raise RuntimeError('Resume refused: training source code changed')
            if restored['runtime_sha256'] != canonical_hash(self.runtime):
                raise RuntimeError('Resume refused: training runtime changed')
            if not self.log.is_file() or sha256_file(self.log) != restored['execution_log_sha256']:
                raise RuntimeError('Resume refused: execution log changed after checkpoint')
            self.counters = dict(restored['counters'])
            self._verify_log()
        elif self.log.exists():
            raise RuntimeError('Refusing to replace an existing execution ledger')
        else:
            self.log.touch(exist_ok=False)
            atomic_json_dump(self.runtime, self.output / 'runtime.json')
            snapshot = self.output / 'training_source'
            root = Path(__file__).resolve().parents[2]
            for rel in self.source['files']:
                dest = snapshot / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / rel, dest)
            atomic_json_dump(self.source, snapshot / 'manifest.json')

    def _append(self, event):
        with self.log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')

    def micro_batch(self, epoch, batch_index, batch_size, loss, step_result=None):
        self.counters['micro_batches'] += 1
        self.counters['sampled_patches'] += int(batch_size)
        event = dict(kind='micro_batch', epoch=epoch, batch_index=batch_index,
                     batch_size=int(batch_size), loss=float(loss), optimizer_updated=False,
                     optimizer_attempted=step_result is not None, overflow=False)
        if step_result is not None:
            event.update(optimizer_updated=bool(step_result['optimizer_updated']),
                         overflow=bool(step_result['overflow']), amp_scale=step_result['new_scale'])
            self.counters['optimizer_attempts'] += 1
            self.counters['optimizer_updates'] += int(event['optimizer_updated'])
            self.counters['skipped_optimizer_updates'] += int(event['overflow'])
        self._append(event)

    def epoch_completed(self, epoch):
        self.counters['scheduler_steps'] += 1
        self.counters['epochs_completed'] = int(epoch)
        self._append(dict(kind='epoch_completed', epoch=int(epoch)))

    def _verify_log(self):
        rows = [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]
        micro = [r for r in rows if r['kind'] == 'micro_batch']
        epochs = [r for r in rows if r['kind'] == 'epoch_completed']
        actual = dict(micro_batches=len(micro), sampled_patches=sum(r['batch_size'] for r in micro),
                      optimizer_attempts=sum(r['optimizer_attempted'] for r in micro),
                      optimizer_updates=sum(r['optimizer_updated'] for r in micro),
                      skipped_optimizer_updates=sum(r['overflow'] for r in micro),
                      scheduler_steps=len(epochs), epochs_completed=epochs[-1]['epoch'] if epochs else 0)
        if actual != self.counters:
            raise RuntimeError('Execution ledger events disagree with checkpoint counters')

    def state(self):
        self._verify_log()
        return dict(counters=dict(self.counters), execution_log_sha256=sha256_file(self.log),
                    source_code_sha256=self.source['sha256'], runtime_sha256=canonical_hash(self.runtime))

    def bind_checkpoint(self, checkpoint):
        checkpoint = Path(checkpoint)
        state = self.state()
        log_copy = checkpoint.with_suffix('.execution.jsonl')
        shutil.copyfile(self.log, log_copy)
        t = self.cfg['training']
        evidence = {**self.counters, 'checkpoint_sha256': sha256_file(checkpoint),
                    'source_config_sha256': self.cfg['_config_sha256'],
                    'training_manifest_sha256': self.manifests['train'],
                    'world_size': 1, 'batch_size_per_gpu': int(t['batch_size_per_gpu']),
                    'gradient_accumulation': int(t.get('gradient_accumulation', 1)),
                    'effective_global_batch_size': int(t['target_global_batch_size']),
                    'optimizer': 'AdamW', 'scheduler': {'name': 'CosineAnnealingLR', 'T_max': int(t['epochs'])},
                    'learning_rate': t['learning_rate'], 'weight_decay': t['weight_decay'],
                    'source_code_sha256': state['source_code_sha256'], 'runtime_sha256': state['runtime_sha256'],
                    'source_log_files': {log_copy.name: sha256_file(log_copy),
                                         'runtime.json': sha256_file(self.output / 'runtime.json'),
                                         'training_source/manifest.json': sha256_file(self.output / 'training_source/manifest.json'),
                                         **{f'training_source/{rel}': digest for rel, digest in self.source['files'].items()}},
                    'fixture': bool(self.cfg['experiment'].get('fixture', False)),
                    'evidence_class': self.cfg['experiment'].get('evidence_class', 'validation_pilot')}
        atomic_json_dump(evidence, checkpoint.with_suffix('.budget.json'))
        return evidence
