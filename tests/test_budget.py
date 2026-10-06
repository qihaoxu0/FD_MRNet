import json
import pytest
import torch

from fdmrnet.engine.budget import TrainingBudgetLedger
from fdmrnet.engine.checkpoint import save_checkpoint, load_checkpoint
from fdmrnet.engine.amp_step import complete_amp_optimizer_step


def config():
    return {'_config_sha256': 'c' * 64, 'experiment': {'fixture': True},
            'training': dict(epochs=2, batch_size_per_gpu=1, gradient_accumulation=1,
                             target_global_batch_size=1, learning_rate=1e-3, weight_decay=.001)}


def test_executed_updates_and_resume_log_binding(tmp_path):
    cfg = config()
    ledger = TrainingBudgetLedger(tmp_path, cfg, {'train': 't' * 64})
    ledger.micro_batch(1, 1, 1, .5, {'optimizer_updated': False, 'overflow': True, 'new_scale': 64.})
    ledger.micro_batch(1, 2, 1, .4, {'optimizer_updated': True, 'overflow': False, 'new_scale': 64.})
    ledger.epoch_completed(1)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    ck = tmp_path / 'epoch_001.pt'
    save_checkpoint(ck, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=1, best_metric=0., config=cfg, manifest_hashes={'train': 't' * 64},
                    training_state=ledger.state())
    first = ledger.bind_checkpoint(ck)
    assert first['micro_batches'] == 2 and first['optimizer_updates'] == 1
    assert first['skipped_optimizer_updates'] == 1
    state = load_checkpoint(ck, model)['training_state']
    resumed = TrainingBudgetLedger(tmp_path, cfg, {'train': 't' * 64}, restored=state)
    resumed.micro_batch(2, 1, 1, .3)
    assert (tmp_path / 'epoch_001.execution.jsonl').read_text().count('\n') == 3
    with pytest.raises(RuntimeError, match='execution log changed'):
        TrainingBudgetLedger(tmp_path, cfg, {'train': 't' * 64}, restored=state)


def test_ledger_rejects_source_runtime_counter_tampering_and_ddp(tmp_path):
    cfg = config()
    ledger = TrainingBudgetLedger(tmp_path, cfg, {'train': 't' * 64})
    state = ledger.state()
    for key, text in [('source_code_sha256', 'source code'), ('runtime_sha256', 'runtime')]:
        changed = {**state, key: '0' * 64}
        with pytest.raises(RuntimeError, match=text):
            TrainingBudgetLedger(tmp_path, cfg, {'train': 't' * 64}, restored=changed)
    changed = {**state, 'counters': {**state['counters'], 'optimizer_updates': 3}}
    with pytest.raises(RuntimeError, match='events disagree'):
        TrainingBudgetLedger(tmp_path, cfg, {'train': 't' * 64}, restored=changed)
    with pytest.raises(RuntimeError, match='one GPU'):
        TrainingBudgetLedger(tmp_path, cfg, {'train': 't' * 64}, world_size=2)


def test_epoch_scheduler_not_advanced_by_micro_step():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    result = complete_amp_optimizer_step(scaler=scaler, optimizer=opt, scheduler=None,
                                         parameters=model.named_parameters(), gradient_clip=1.)
    assert result['optimizer_updated'] and sched.last_epoch == 0
    sched.step()
    assert sched.last_epoch == 1


def test_collector_recounts_budget_events_instead_of_trusting_summary(tmp_path):
    import sys
    from pathlib import Path
    from fdmrnet.config import sha256_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from collect_results import training_budget_policy
    cfg = config()
    train = tmp_path / 'train.csv'
    train.write_text('subject\ncase_a\n')
    cfg['data'] = dict(train_manifest=str(train), train_samples_per_epoch=1, patch_size_dhw=[8,16,16])
    cfg['training']['record_actual_budget'] = True
    ledger = TrainingBudgetLedger(tmp_path / 'run', cfg, {'train': sha256_file(train)})
    for epoch in (1,2):
        ledger.micro_batch(epoch, 1, 1, .5, {'optimizer_updated': True, 'overflow': False, 'new_scale': 64.})
        ledger.epoch_completed(epoch)
    ck = ledger.output / 'epoch_002.pt'
    ck.write_bytes(b'technical checkpoint hash fixture only')
    evidence = ledger.bind_checkpoint(ck)
    path = ck.with_suffix('.budget.json')
    def audit():
        return dict(checkpoint_sha256=sha256_file(ck), training_budget_evidence_path=str(path),
                    training_budget_evidence_sha256=sha256_file(path))
    assert training_budget_policy(cfg, audit(), False) != 'unknown_diagnostic'
    evidence['optimizer_updates'] = 1
    path.write_text(json.dumps(evidence))
    with pytest.raises(RuntimeError, match='counters disagree'):
        training_budget_policy(cfg, audit(), False)
