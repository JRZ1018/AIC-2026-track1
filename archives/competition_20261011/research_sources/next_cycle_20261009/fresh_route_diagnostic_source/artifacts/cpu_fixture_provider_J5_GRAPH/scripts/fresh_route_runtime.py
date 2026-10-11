"""Fresh matched-runtime, trainable-count and H1 assertions; no new optimizer logic."""
from __future__ import annotations
import json,os,time,uuid
from pathlib import Path

REMOTE_ROOT='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009'
INITIAL_FREE_BYTES=13013950464
RUNTIME_ENV={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}

def runtime_values(torch):
    return dict(environment={k:os.environ.get(k) for k in RUNTIME_ENV},
                actual_torch_threads=torch.get_num_threads(),
                actual_torch_interop_threads=torch.get_num_interop_threads())

def validate_runtime(value):
    if value.get('environment')!=RUNTIME_ENV or value.get('actual_torch_threads')!=4:
        raise RuntimeError('fresh route requires OMP4/MKL4/CUBLAS4096:8 and actual torch threads4')

def require_runtime(root,config):
    import torch
    from p1_contract import sha,assets
    if os.name!='posix' or Path(root).resolve()!=Path(REMOTE_ROOT).resolve():
        raise RuntimeError('fresh route executes only on the exact dedicated data-disk root')
    value=runtime_values(torch);validate_runtime(value)
    target=Path(root)/'artifacts/runtime_receipts';target.mkdir(parents=True,exist_ok=True)
    receipt=dict(schema='aictrack1.fresh_route.runtime.v1',arm=config.stem.removeprefix('p1_'),selected_route=assets()['selected_route'],pid=os.getpid(),
        config_sha256=sha(config),source_manifest_sha256=sha(Path(root)/'source_manifest.json'),
        collected_at_unix=time.time(),test_access=False,**value)
    path=target/f'{config.stem.removeprefix("p1_")}_{os.environ["P1_ATTEMPT_ID"]}.json'
    with path.open('x',encoding='utf-8') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')

def assert_route_head(model,head,reference,raw):
    import torch
    active=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    joint=raw["training"]["head_learning_rate"]>0
    if len(active)!=(154 if joint else 152) or sum(p.numel() for _,p in active)!=(88233966 if joint else 87849216):
        raise RuntimeError('fresh route requires exact 152/87849216 or joint154/88233966 trainables')
    if joint:
        if not all(p.requires_grad for p in head.parameters()):raise RuntimeError('fresh joint head not trainable')
        return
    if any(name.startswith('classifier.') for name,_ in active) or any(p.requires_grad for p in head.parameters()):
        raise RuntimeError('fresh C/S head must stay frozen from the new legal fold H1')
    current=head.state_dict()
    if set(current)!=set(reference) or any(not torch.equal(current[k].detach().cpu(),v) for k,v in reference.items()):
        raise RuntimeError('fresh C/S new fold H1 head changed, including after resume')

def clip_diagnostics(norms,maximum,successful_updates,amp_skips):
    import torch
    values=norms.detach().to(device='cpu',dtype=torch.float32)
    finite=values[torch.isfinite(values)]
    # This is the same clamped multiplier used by torch clip_grad_norm_.
    coefficients=(float(maximum)/(finite+1e-6)).clamp(max=1.)
    quantiles=[0.,.1,.25,.5,.75,.9,1.]
    distribution=None if not len(coefficients) else {
        str(q):float(v) for q,v in zip(quantiles,torch.quantile(coefficients,torch.tensor(quantiles)))}
    return dict(clip_coefficient_quantiles=distribution,
        clip_coefficient_definition='min(1, max_norm/(pre_clip_total_l2_norm+1e-6))',
        clip_coefficient_finite_norm_count=len(finite),
        clip_coefficient_nonfinite_excluded=int((~torch.isfinite(values)).sum()),
        amp_successful_optimizer_updates=int(successful_updates),amp_skipped_optimizer_updates=int(amp_skips),
        amp_step_observation='public GradScaler before/after scale; decrease means skipped update; no algorithm change',
        epoch_scheduler_policy='original FV2 once per epoch, unchanged by AMP skips')

def bind_budget(root,*,start,deadline,initial,diskmax,minimum):
    if initial!=INITIAL_FREE_BYTES or minimum<3*1024**3 or diskmax>8*1024**3 or diskmax<=0:
        raise ValueError('fresh cumulative original data baseline/min3GiB/max8GiB required')
    now=time.time()
    if start!=1791497972.440027 or deadline!=1791526772.440027 or now>=deadline:
        raise ValueError('root must specify original fresh phase start/deadline, no reset')
    expected=dict(schema='aictrack1.fresh_route.budget.v1',phase_start_unix=start,deadline_unix=deadline,
                  initial_free_bytes=initial,disk_max_bytes=diskmax,min_free_bytes=minimum)
    path=Path(root)/'artifacts/queues/fresh_route_budget.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=expected:
            raise ValueError('fresh phase budget already bound; do not reset deadline or free baseline')
    else:
        with path.open('x',encoding='utf-8') as handle:json.dump(expected,handle,indent=2);handle.write('\n')
    return expected
