"""Independent final inference contracts; no test image reads in admission."""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_SOURCE_SHA = json.loads((ROOT/'source_manifest.json').read_text()).get('full_training_source_sha256') if (ROOT/'source_manifest.json').is_file() else None
FULL_ROOT = Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_full_20261009')
REMOTE_ROOT = Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_final_20261009')
LOCK = Path('/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json')
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA = '1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA = '811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
TEST_SHA = '576ff0f69e0a104ca5a0af7f190628074b572102c6bc8daf8cb653fb08054201'
TEST_MANIFEST = Path('/root/autodl-tmp/AICtrack1_p21_full20_20261001/artifacts/audits/stage2/test/samples.csv')
OFFICIAL_DATA_ROOT = '/root/autodl-tmp/AICtrack1/DataSetStage2'
P7_SHA = '8122c9c96fb80823db9d19edb403d8d147d26a32c43b7b48fe7cca9db311d703'
VIEW = 'native336_center_and_exact_flip_plus_resize_short_side384_center_crop336_and_exact_flip_equal_four_logits_single_checkpoint'
RUNTIME = {'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
GIB = 1024 ** 3
FULL_H1_SHA='fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605'
def valid_digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def require(ok, reason):
    if not ok: raise ValueError(reason)

def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write('\n')

def verify_source(root=ROOT, expected=None):
    root = Path(root).resolve(); path = root/'source_manifest.json'
    if expected is not None: require(sha(path)==expected, 'inference source manifest SHA differs')
    value = read(path)
    require(value.get('schema')=='aictrack1.fresh_route.final_inference.source.v1' and value.get('status')=='bound_actual_full20_checkpoint' and value.get('full_training_source_sha256')==FULL_SOURCE_SHA and value.get('selected_route') in ('S','J5_GRAPH'), 'wrong final source identity')
    require(valid_digest(value.get('full_training_source_sha256')) and valid_digest(value.get('final_sha256')) and valid_digest(value.get('last_sha256')) and valid_digest(value.get('confirmation_sha256')),'actual FULL/checkpoint/fresh-confirmation SHA required')
    binding=read(root/'inputs/actual_full_binding.json')
    for key in ('selected_route','selected_arm','full_training_source_sha256','confirmation_sha256','final_sha256','last_sha256'):require(binding.get(key)==value.get(key),'actual binding/source alias '+key)
    for name, digest in value['files'].items():
        rel = Path(name); target = root/rel
        require(not rel.is_absolute() and '..' not in rel.parts and not target.is_symlink() and target.resolve().is_relative_to(root), 'unsafe source member')
        require(sha(target)==digest, 'source file drift: '+name)
    return value

def verify_full_source(full):
    full = Path(full).resolve()
    require(sha(full/'source_manifest.json')==FULL_SOURCE_SHA, 'unaccepted FULL source')
    value = read(full/'source_manifest.json')
    require(value.get('schema')=='aictrack1.fresh_route.full.training.source.v1' and value.get('status')=='bound', 'wrong FULL schema')
    require(value.get('template_manifest_sha256')==sha(ROOT/'references/UPSTREAM_FULL_TEMPLATE.json'),'unreviewed actual FULL producer template')
    for name, digest in value['files'].items():
        rel=Path(name); p=full/rel
        require(not rel.is_absolute() and '..' not in rel.parts and not p.is_symlink() and p.resolve().is_relative_to(full) and sha(p)==digest, 'FULL source drift: '+name)
    critical=[name for name in value['files'] if name.startswith('src/aictrack1/models/')]+['src/aictrack1/engine/peft.py','src/aictrack1/evaluation/four_view.py','src/aictrack1/data/transforms.py','src/aictrack1/submission/package.py']
    for name in critical:require(sha(ROOT/name)==value['files'][name],'FULL/inference model/view implementation mismatch: '+name)
    return value

def runtime(torch):
    actual = {key:os.environ.get(key) for key in RUNTIME}
    require(actual==RUNTIME and torch.get_num_threads()==4, 'actual inference CPU thread/runtime policy differs')
    return actual | {'torch_num_threads':torch.get_num_threads(), 'torch_num_interop_threads':torch.get_num_interop_threads(), 'python':sys.version, 'torch':str(torch.__version__)}

def validate_metrics(metrics, last, final, *, config_sha, final_sha, last_sha, head_sha, arm):
    require(arm in ('S','J5'), 'only positively selected typed FULL arm is admissible')
    require(metrics.get('training_scope')=='full_train_fixed_epochs' and metrics.get('train_count')==148695 and metrics.get('validation_count')==0, 'FULL148695/validation0 metrics required')
    require(metrics.get('early_stop') is None and metrics.get('initial_validation') is None and metrics.get('best_top1') is None, 'partial/best/validation selected training forbidden')
    require(metrics.get('selected_checkpoint')=='final.pt' and metrics.get('selection_rule')=='fixed final epoch 20; no validation selection', 'fixed final20 metrics required')
    require(metrics.get('selected_checkpoint_sha256')==metrics.get('final_checkpoint_sha256')==final_sha and metrics.get('last_checkpoint_sha256')==last_sha, 'actual metrics/final/last hashes differ')
    require(metrics.get('config_sha256')==last.get('config_sha256')==final.get('config_sha256')==config_sha, 'actual original config binding differs')
    require(metrics.get('head_initializer',{}).get('sha256')==final.get('head_initializer_sha256')==head_sha, 'FULL H1 initializer binding differs')
    require(last.get('epoch')==20 and last.get('training_scope')==final.get('training_scope')=='full_train_fixed_epochs', 'last must be complete FULL epoch20')
    require(last.get('best_state') is None and last.get('initial_validation') is None and not last.get('trainable_ema'), 'unexpected best/EMA recovery state')
    require(final.get('selection_rule')=='fixed final epoch 20; no validation selection' and final.get('normalize_features') is True, 'final protocol differs')
    for value in (metrics, last, final):
        require(value.get('visual_input_resolution')==336 and value.get('position_interpolation')=='bicubic_7x7_to_10x10', 'resolution/interpolation differs')
    require(last.get('mode')==final.get('mode')=='full_visual', 'wrong visual scope')
    require([r.get('epoch') for r in metrics.get('history',[])]==list(range(1,21)) and [r.get('epoch') for r in last.get('history',[])]==list(range(1,21)), 'incomplete20 history')
    require(metrics['history']==last['history'] and all('validation' not in row and 'ema_validation' not in row for row in last['history']), 'FULL history differs or contains validation')
    for value in (last,final):
        for key, expected in [('competition_stage','stage2'),('clip_weight_sha256',CLIP_SHA),('train_manifest_sha256',MANIFEST_SHA),('class_mapping_sha256',MAPPING_SHA)]:
            require(value.get(key)==expected, 'official provenance differs: '+key)
    require({'optimizer','scheduler','scaler','python_rng_state','numpy_rng_state','torch_rng_state','cuda_rng_state_all','confidence_weighting'}.issubset(last), 'incomplete recovery state')
    require(metrics.get('trainable_total_parameters')==(87849216 if arm=='S' else 88233966), 'metrics head policy differs')
    route='S' if arm=='S' else 'J5_GRAPH'
    expected_head='legal_full_H1_frozen' if arm=='S' else 'joint_fixed_last_state'
    require(last.get('fresh_route_full_arm')==arm and last.get('selected_route')==route and last.get('fresh_route_full_source_manifest_sha256')==FULL_SOURCE_SHA, 'typed FULL last source/route differs')
    for value in (last,final,metrics):
        require(value.get('fresh_route_full_arm')==arm and value.get('selected_route')==route and value.get('fresh_route_full_source_manifest_sha256')==FULL_SOURCE_SHA and value.get('classifier_head_source')==expected_head and value.get('classifier_head_restored_equal') is (arm=='S'), 'typed FULL checkpoint/metrics head source differs')
    require(metrics.get('base_loss')=={'kind':'sce','alpha':(.5 if arm=='S' else .1),'beta':1.,'log_zero':-4.}, 'typed FULL SCE recipe differs')
    return True

def validate_joint_tensors(last, final, torch, *, route='J5_GRAPH', expected_visual=152, expected_numel=None):
    require(route in ('S','J5_GRAPH'),'unknown final route')
    state=last.get('trainable_state',{});export=final.get('trainable_state',{});joint=route=='J5_GRAPH'
    head={'classifier.linear.weight','classifier.linear.bias'} if joint else set()
    require(set(state)==set(export) and len(state)==expected_visual+len(head),'final/last incomplete trainable state')
    require({n for n in state if n.startswith('classifier.')}==head and all(n.startswith('visual.') or n in head for n in state),'partial/unknown classifier state')
    if expected_numel is None:expected_numel=88233966 if joint else 87849216
    require(sum(t.numel() for t in state.values())==expected_numel,'parameter count differs')
    if joint:require(state['classifier.linear.weight'].shape==(750,512) and state['classifier.linear.bias'].shape==(750,),'joint head shape differs')
    for name,value in state.items():
        require(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and torch.isfinite(value).all().item() and torch.equal(value,export[name]),'final/last tensor differs or nonfinite: '+name)
    return state


def restore_joint(visual,head,state,torch,*,route='J5_GRAPH',expected_visual=152):
    from aictrack1.engine.peft import configure_classifier_trainability_from_state,load_trainable_state_dict
    from aictrack1.models.peft import PEFTImageClassifier
    require(route in ('S','J5_GRAPH'),'unknown restoration route')
    initial={f'classifier.{n}':p.detach().cpu().clone() for n,p in head.named_parameters()}
    joint=configure_classifier_trainability_from_state(head,state)
    require(joint==(route=='J5_GRAPH'),'typed frozen/learned head state mismatch')
    model=PEFTImageClassifier(visual,head,normalize_features=True)
    names={n for n,p in model.named_parameters() if p.requires_grad}
    require(len(names)==expected_visual+(2 if joint else 0) and names==set(state),'model trainable scope differs')
    load_trainable_state_dict(model,state)
    for n,p in model.named_parameters():
        if n in names:require(torch.equal(p.detach().cpu(),state[n].cpu()),'CPU restored parameter differs: '+n)
        elif n.startswith('classifier.'):require(torch.equal(p.detach().cpu(),initial[n]),'legal full H1 frozen head changed')
    delta={n:float((dict(model.named_parameters())[n].detach().cpu()-initial[n]).abs().max()) for n in sorted(initial)}
    return model,delta


def restore_optimizer(model,last,torch,head_lr):
    visual=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('visual.')]
    head=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('classifier.')]
    saved=last['optimizer'];groups=saved['param_groups'];counts=[len(visual)]+([2] if head_lr>0 else [])
    require([len(g['params']) for g in groups]==counts and len(head)==(2 if head_lr>0 else 0),'AdamW152/154 group order differs')
    base=[1.75e-6]+([head_lr] if head_lr>0 else [])
    require(last['scheduler']['T_max']==last['scheduler']['last_epoch']==20 and last['scheduler']['base_lrs']==base,'cosine20/LR schedule differs')
    pg=[dict(params=visual,lr=1.75e-6)]+([dict(params=head,lr=head_lr)] if head_lr>0 else [])
    opt=torch.optim.AdamW(pg,weight_decay=.01);opt.load_state_dict(saved)
    require(len(opt.state)==sum(counts),'AdamW moments incomplete')
    for params,group in zip([visual]+([head] if head_lr>0 else []),groups,strict=True):
        for param,key in zip(params,group['params'],strict=True):
            loaded=opt.state[param];entry=saved['state'][key]
            require(set(loaded)==set(entry),'optimizer moment keys differ')
            require(loaded['exp_avg'].shape==loaded['exp_avg_sq'].shape==param.shape,'optimizer moment shape differs')
            require(float(loaded['step'])>0 and bool((loaded['exp_avg_sq']>=0).all()),'optimizer update/moment population invalid')
            for name,value in loaded.items():
                if isinstance(value,torch.Tensor):require(torch.equal(value,entry[name]) and torch.isfinite(value).all().item(),'optimizer state differs/nonfinite')
                else:require(value==entry[name],'optimizer scalar differs')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20);scheduler.load_state_dict(last['scheduler'])
    require(scheduler.state_dict()==last['scheduler'],'scheduler restore differs')
    scaler=torch.amp.GradScaler('cpu');scaler.load_state_dict(last['scaler']);require(scaler.state_dict()==last['scaler'],'scaler restore differs')
    return opt,scheduler,scaler


def validate_dataset_override(original, override):
    import copy
    expected=copy.deepcopy(original); expected['dataset']['sources']['test']={'root':'test','label_source':'none'}
    expected['dataset']['dataset_root']=OFFICIAL_DATA_ROOT
    require(override==expected, 'inference overlay may only declare the pinned official data root/test source')
    require(override['dataset']['allow_truncated_images'] is False, 'changed image decode policy')
    return override

def validate_unlabeled_test_rows(rows, *, expected_count=37444):
    require(len(rows)==expected_count, 'official test count differs')
    names=[]; ids=[]
    for row in rows:
        rel=Path(row['relative_path'])
        require(not rel.is_absolute() and len(rel.parts)==1 and '..' not in rel.parts and rel.suffix.lower()=='.jpg', 'unsafe/non-flat test filename')
        require(row.get('source_split')=='test' and not row.get('internal_class_index') and not row.get('official_class_id'), 'test labels or wrong source forbidden')
        names.append(rel.name); ids.append(row['sample_id'])
    require(len(set(names))==len(names) and len(set(ids))==len(ids), 'duplicate test filename/ID')
    return names

def read_test_rows(path):
    require(sha(path)==TEST_SHA, 'official test manifest SHA differs')
    with Path(path).open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
    validate_unlabeled_test_rows(rows); return rows

def audit_package(output, expected, mapping, frozen):
    from aictrack1.submission import SubmissionSpec,validate_submission_csv,validate_submission_zip
    output=Path(output); spec=SubmissionSpec(include_header=False,filename_mode='basename',official_class_id_pattern=r'\d{4}')
    allowed=[e.official_class_id for e in mapping.entries]
    validate_submission_csv(output/'pred_results.csv',expected,allowed,spec=spec)
    validate_submission_zip(output/'pred_results.zip',expected,allowed,spec=spec)
    with zipfile.ZipFile(output/'pred_results.zip') as archive:
        require(archive.read('pred_results.csv')==(output/'pred_results.csv').read_bytes(), 'ZIP/CSV bytes differ')
    m=read(output/'submission_manifest.json'); meta=m['run_metadata']
    require(m['record_count']==len(expected)==37444 and m['class_mapping_sha256']==mapping.digest()==MAPPING_SHA, 'package mapping/count differs')
    require(meta['checkpoint_sha256']==frozen['final_sha256'] and meta['full_training_source_sha256']==FULL_SOURCE_SHA and meta['classifier_head_source']==frozen['classifier_head_source'], 'package checkpoint/head/source differs')
    require(meta['inference_view_protocol']==VIEW and meta['balanced_prior']['temperature_multiplier']==.5 and meta['balanced_prior']['bias_strength']==.75 and meta['balanced_prior_selection']['report_sha256']==P7_SHA, 'package changed frozen inference recipe')
    require(meta['test_manifest_sha256']==TEST_SHA and meta['test_record_count']==37444, 'package test manifest differs')
    require(m['csv_sha256']==sha(output/'pred_results.csv') and m['zip_sha256']==sha(output/'pred_results.zip'), 'package byte hashes differ')
    require(meta.get('selected_route')==frozen['selected_route'],'submission route differs')
    if frozen['selected_route']=='J5_GRAPH':require(meta.get('fixed_graph',{}).get('parameters')=={'k':10,'lambda':.3,'iterations':3,'probability_temperature':1.,'protection_raw_margin':math.log(4.)},'fixed graph submission policy differs')
    return {n:sha(output/n) for n in ('pred_results.csv','pred_results.zip','submission_manifest.json')}

def resource_check(root, policy, *, reserve=0):
    free=shutil.disk_usage(root).free
    require(time.time()<policy['deadline_unix'] and policy['initial_free_bytes']>0 and policy['min_free_bytes']>=3*GIB, 'invalid/expired inference resource boundary')
    require(free-reserve>=policy['min_free_bytes'] and policy['initial_free_bytes']-free+reserve<=policy['disk_max_bytes'], 'inference disk reserve exceeded')
    if sys.platform=='linux':
        memory={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1);memory[key]=int(value.strip().split()[0])*1024
        require(memory['MemAvailable']>=policy['min_available_memory_bytes'], 'available RAM guard triggered')
    return free
