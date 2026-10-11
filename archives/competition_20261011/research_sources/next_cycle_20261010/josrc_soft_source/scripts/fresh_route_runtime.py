"""Fresh matched-runtime, trainable-count and H1 assertions; no new optimizer logic."""
from __future__ import annotations
import json,os,time,uuid
from pathlib import Path

from p1_contract import REMOTE_ROOT as EXEC_ROOT
REMOTE_ROOT=str(EXEC_ROOT)
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


def budget(root):
    from p1_contract import sha,assets,REMOTE_ROOT
    b=json.loads((Path(root)/'inputs/budget.json').read_text(encoding='utf-8'))
    if b.get('schema') not in ('aictrack1.supervision_joint.budget.v1','aictrack1.josrc_inspired.budget.v1') or b.get('root_authorized') is not True or b.get('test_access') is not False:raise ValueError('actual root-authorized B/profile budget required')
    expected={'initial_free_bytes':13013950464,'capacity_increase_bytes':32212254720,'adjusted_initial_free_bytes':45226205184,'data_total_bytes':128849018880,'min_free_bytes':3221225472,'atomic_reserve_bytes':2147483648,'final_gpu_reserve_hours':4}
    if any(b.get(k)!=v for k,v in expected.items()) or not isinstance(b.get('disk_max_bytes'),int) or not 21474836480<=b['disk_max_bytes']<=32212254720 or not b['phase_start_unix']<b['deadline_unix']:raise ValueError('same cumulative ledger/actual stage clock/final4h reserve required')
    payload=assets(root)
    if payload.get('execution_root')!=str(REMOTE_ROOT) or payload.get('actual_budget_sha256')!=sha(Path(root)/'inputs/budget.json'):raise ValueError('actual B dedicated root/budget source binding')
    return b

def resource_check(root,reserve=0):
    import shutil
    b=budget(root);disk=shutil.disk_usage(root)
    if disk.total!=b['data_total_bytes'] or not b['phase_start_unix']<=time.time()<b['deadline_unix'] or disk.free-reserve<b['min_free_bytes'] or b['adjusted_initial_free_bytes']-disk.free+reserve>b['disk_max_bytes']:raise RuntimeError('B stage actual deadline/cumulative disk/reserve exceeded')
    return b
