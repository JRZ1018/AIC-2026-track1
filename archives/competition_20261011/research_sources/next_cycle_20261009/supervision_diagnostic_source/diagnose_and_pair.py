"""Explicit root entry: one S fixed-last20 four-view pass, then CPU frozen P7.

There is no scheduler, connection, background wait, training or recipe selection.
The raw diagnostic obtains the repository-wide TrainingLease itself.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def commands(args):
    output = ROOT / 'artifacts/diagnostics/P1_S'
    diagnostic = [sys.executable, str(ROOT/'run_supervision_diagnostic.py'),
        '--training-root', str(args.training_root), '--arm', 'S',
        '--expected-last-sha256', args.expected_last_sha256,
        '--deadline-unix', str(args.deadline_unix), '--hard-seconds', str(args.hard_seconds),
        '--workers', str(args.workers), '--min-free-gib', str(args.min_free_gib), '--run']
    if args.save_features:
        diagnostic.append('--save-features')
    compare = [sys.executable, str(ROOT/'compare_supervision.py'),
        '--candidate-bundle', str(output/'fused_logits.pt'),
        '--candidate-identity', str(output/'input_identity.json'),
        '--control-bundle', str(args.control_bundle), '--control-identity', str(args.control_identity),
        '--output', str(ROOT/'artifacts/paired/P1_S'), '--threads', '4']
    return diagnostic, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', required=True, type=Path)
    parser.add_argument('--expected-last-sha256')
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--hard-seconds', type=int, default=1800)
    parser.add_argument('--workers', type=int, choices=(0,4), default=4)
    parser.add_argument('--min-free-gib', type=float, default=3.)
    parser.add_argument('--save-features', action='store_true')
    parser.add_argument('--control-bundle', type=Path)
    parser.add_argument('--control-identity', type=Path)
    parser.add_argument('--dependency-check', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.dependency_check:
        if args.run:
            raise ValueError('dependency-check cannot execute inference')
        return subprocess.run([sys.executable,str(ROOT/'run_supervision_diagnostic.py'),
            '--training-root',str(args.training_root),'--dependency-check'],check=False).returncode
    if (not args.run or args.control_bundle is None or args.control_identity is None
            or args.deadline_unix is None or args.expected_last_sha256 is None
            or not 60 <= args.hard_seconds <= 3600):
        raise ValueError('explicit bounded run, last SHA, deadline and pinned C inputs required')
    if (os.name != 'posix' or ROOT != Path('/root/autodl-tmp/AICtrack1_next_cycle_supervision_diagnostic_20261009').resolve()
            or args.min_free_gib < 3.):
        raise ValueError('literal data-disk production root and minimum3GiB required')
    diagnostic, compare = commands(args)
    # Check the control on CPU before permitting the expensive one-model forward.
    import torch
    torch.set_num_threads(4)
    from supervision_diag_contract import load_cache, verify_owned_sources, CONTROL_CACHE_SHA, CONTROL_LAST_SHA
    verify_owned_sources()
    control = load_cache(args.control_bundle,args.control_identity)
    if control['identity']['bundle_sha256'] != CONTROL_CACHE_SHA or control['identity']['checkpoint_sha256'] != CONTROL_LAST_SHA:
        raise ValueError('requires accepted C fixed-last20; best and last8 are forbidden')
    del control
    if args.deadline_unix-time.time() < args.hard_seconds+120:
        raise TimeoutError('one complete diagnostic and CPU pairing must fit the existing phase deadline')
    # No detached child here: an operator must retain this bounded foreground entry.
    # Child owns its fixed wall alarm and canonical lease; failure preserves its output.
    result = subprocess.run(diagnostic,check=False)
    if result.returncode:
        return result.returncode
    remaining = args.deadline_unix-time.time()
    if remaining < 60:
        raise TimeoutError('insufficient remaining CPU paired budget; complete cache is retained')
    return subprocess.run(compare,check=False,timeout=remaining).returncode


if __name__ == '__main__':
    raise SystemExit(main())
