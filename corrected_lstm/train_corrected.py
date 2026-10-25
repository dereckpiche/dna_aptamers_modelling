"""Local exploratory run; never writes to GitHub. Run with --resume after interruption."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SPEC = dict(seed=20261025, hidden_size=64, layers=2, dropout=0.6,
            learning_rate=0.001, batch_size=512, max_epochs=15,
            patience=4, train_fraction=0.8, validation_fraction=0.1,
            input_shape="batch,30,4", encoding="ATCG", threads=2)


class StrandLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, SPEC['hidden_size'], SPEC['layers'],
                            dropout=SPEC['dropout'], batch_first=True)
        self.fc = nn.Linear(SPEC['hidden_size'], 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (30, 4):
            raise ValueError('Expected independent strands shaped [batch,30,4]')
        # No state is carried between calls or between strands in the batch.
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def atomic_save(obj, name):
    torch.save(obj, ROOT / (name + '.tmp'))
    os.replace(ROOT / (name + '.tmp'), ROOT / name)


def independence_tests(model):
    model.eval()
    x = torch.nn.functional.one_hot(torch.randint(4, (5, 30)), 4).float()
    with torch.no_grad():
        p = model(x)
        alone = model(x[:1])[0]
        order = torch.tensor([3, 0, 4, 1, 2])
        shuffled = model(x[order])
        changed = x.clone(); changed[1:] = x[1:].flip(1)
        unchanged = model(changed)[0]
    assert torch.allclose(p[0], alone, atol=1e-6)
    assert torch.allclose(shuffled, p[order], atol=1e-6)
    assert torch.allclose(p[0], unchanged, atol=1e-6)
    try:
        model(x.reshape(5, 120))
        raise AssertionError('Flattened input should be rejected')
    except ValueError:
        pass
    return {'alone_vs_batch': True, 'batch_permutation': True,
            'other_strands_do_not_change_prediction': True, 'reject_flat_input': True}


def evaluate(model, x, y, indices):
    model.eval()
    sse = 0.0
    with torch.inference_mode():
        for ix in indices.split(SPEC['batch_size']):
            pred = model(torch.nn.functional.one_hot(x[ix].long(), 4).float())
            sse += (pred.double() - y[ix].double()).square().sum().item()
    return sse / len(indices)


def main(resume=False):
    torch.set_num_threads(SPEC['threads'])
    random.seed(SPEC['seed']); np.random.seed(SPEC['seed']); torch.manual_seed(SPEC['seed'])
    torch.use_deterministic_algorithms(True)
    if (ROOT / 'final_results.json').exists():
        print('Completed run already exists; refusing to reopen the test set.', flush=True)
        return
    raw = (ROOT / 'strandenergylist.json').read_bytes()
    rows = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert len(rows) == 1000000 and all(len(s) == 30 and set(s) <= set('ATCG') and np.isfinite(v) for s, v in rows)
    sequences = [r[0] for r in rows]
    assert len(set(sequences)) == len(sequences), 'Duplicate strands require grouped splitting'
    lookup = np.zeros(256, dtype=np.uint8)
    for i, b in enumerate(b'ATCG'): lookup[b] = i
    x = torch.from_numpy(lookup[np.frombuffer(''.join(sequences).encode(), dtype=np.uint8)].reshape(-1, 30).copy())
    y = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    del raw, rows, sequences
    permutation = torch.randperm(len(y), generator=torch.Generator().manual_seed(SPEC['seed']))
    train, valid, test = permutation[:800000], permutation[800000:900000], permutation[900000:]
    np.savez_compressed(ROOT / 'split_indices.npz', train=train.numpy(), validation=valid.numpy(), test=test.numpy())
    metadata = dict(spec=SPEC, dataset_sha256=digest, observations=len(y),
                    split_sizes=[len(train), len(valid), len(test)], unique_strands=len(y),
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    source='https://github.com/dereckpiche/dna_aptamers_modelling/blob/main/src/data/strandenergylist.json',
                    original_results=dict(MLP_report=0.94, LSTM_report=0.91,
                                          Transformer_report=0.15, Larger_Transformer_best_report=0.12),
                    limitations=['New split; original split indices unavailable.',
                                 'Smaller hidden dimension than original; not a controlled architecture comparison.',
                                 'CPU fresh training, not continuation of incompatible old weights.',
                                 'One seed; simulated NUPACK labels, not experimental binding measurements.'])
    (ROOT / 'run_spec.json').write_text(json.dumps(metadata, indent=2))
    model = StrandLSTM()
    checks = independence_tests(model)
    opt = torch.optim.Adam(model.parameters(), lr=SPEC['learning_rate'])
    history = []; best = float('inf'); best_epoch = 0; start = 1
    if resume and (ROOT / 'latest.pt').exists():
        state = torch.load(ROOT / 'latest.pt', weights_only=True)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng']); history = state['history']
        best = state['best']; best_epoch = state['best_epoch']; start = state['epoch'] + 1
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()), tests=checks, spec=SPEC)), flush=True)
    for epoch in range(start, SPEC['max_epochs'] + 1):
        begin = time.time(); model.train(); sse = 0.0
        order = train[torch.randperm(len(train))]
        for batch_number, ix in enumerate(order.split(SPEC['batch_size'])):
            inp = torch.nn.functional.one_hot(x[ix].long(), 4).float()
            opt.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = (pred - y[ix]).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            sse += loss.item() * len(ix)
            if (batch_number + 1) % 400 == 0:
                print(json.dumps(dict(epoch=epoch, processed=(batch_number+1)*SPEC['batch_size'],
                                      running_training_mse=sse/((batch_number+1)*SPEC['batch_size']),
                                      seconds=round(time.time()-begin, 1))), flush=True)
        vmse = evaluate(model, x, y, valid)
        record = dict(epoch=epoch, training_mse=sse/len(train), validation_mse=vmse,
                      seconds=time.time()-begin)
        history.append(record)
        if vmse < best:
            best = vmse; best_epoch = epoch
            atomic_save({'model': model.state_dict(), 'epoch':epoch}, 'best.pt')
        atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=torch.get_rng_state(),
                         history=history, best=best, best_epoch=best_epoch, epoch=epoch), 'latest.pt')
        (ROOT / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        if epoch - best_epoch >= SPEC['patience']: break
    state = torch.load(ROOT / 'best.pt', weights_only=True)
    model.load_state_dict(state['model'])
    tests = independence_tests(model)
    mse = evaluate(model, x, y, test)
    mean = y[train].double().mean().item()
    baseline = (y[test].double()-mean).square().mean().item()
    result = dict(test_mse=mse, test_rmse=mse**0.5, validation_mse=best,
                  selected_epoch=best_epoch, epochs_completed=history[-1]['epoch'],
                  test_observations=len(test), training_mean_test_mse=baseline,
                  tests=tests, parameters=sum(p.numel() for p in model.parameters()),
                  total_epoch_seconds=sum(r['seconds'] for r in history))
    (ROOT / 'final_results.json').write_text(json.dumps(result, indent=2))
    print('FINAL ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--resume', action='store_true')
    main(parser.parse_args().resume)
