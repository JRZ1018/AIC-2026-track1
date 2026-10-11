"""Bound joint-J5 loss exploration: actual assets/configs, no old-fold admission clock."""
from __future__ import annotations
import copy,hashlib,json,math
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_supervision_joint_r3_20261010')
ARMS={'A3':.3,'A5':.5}
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def assets(root=ROOT):
    root=Path(root);p=json.loads((root/'inputs/p1_assets.json').read_text(encoding='utf-8'))
    require(p.get('schema')=='aictrack1.fresh_route.assets.v1' and p.get('competition_stage')=='stage2','actual Stage2 asset snapshot required')
    require(p.get('selected_route')=='J5_GRAPH' and p.get('selected_arm')=='J5' and p.get('test_access') is False,'joint J5 official train-only assets required')
    require(p.get('train_count')==118958 and p.get('validation_count')==29737 and p.get('num_classes')==750 and p.get('training_seed')==20261008,'actual exploration rows/seed required')
    require(p.get('canonical_lock')=='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json','canonical queue only')
    expected={'clip':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af','h1':'e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be','split':'57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'}
    for name,digest in expected.items():require(p['assets'][name]['sha256']==digest,'asset pin '+name)
    actual=json.loads((root/'inputs/h1_actual_CPU_complete.json').read_text(encoding='utf-8'))
    require(p['h1_row_binding']==actual['producer_receipt']['row_binding'] and p['h1_row_binding']['counts']=={'train':118958,'validation':29737} and p['h1_row_binding']['validation_used_for_optimization_or_selection'] is False,'legal heldout H1 row binding')
    return p
def expected_config(arm,p=None,root=ROOT):
    require(arm in ARMS,'only registered alpha .3/.5')
    raw=copy.deepcopy(yaml.safe_load((Path(root)/'inputs/baseline_J5.yaml').read_text(encoding='utf-8'))['experiment'])
    raw['base_loss']['alpha']=ARMS[arm];raw['run_id']=f'20261010-supervision-joint-{arm}-fixed20';raw['output_dir']=f'../../artifacts/experiments/JOINT_{arm}'
    require(raw['training']=={'epochs':20,'batch_size':64,'workers':8,'prefetch_factor':2,'persistent_workers':True,'seed':20261008,'visual_learning_rate':1.75e-6,'head_learning_rate':5e-6,'weight_decay':.01,'gradient_clip_norm':1.},'unchanged real J5 training recipe')
    return raw
def validate_config(raw,arm,p):require(raw==expected_config(arm,p),'only alpha/runid/output may differ from actual J5')
def verify_files(root=ROOT):
    root=Path(root).resolve();m=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    require(m.get('schema')=='aictrack1.supervision_joint.source.v1' and m.get('status')=='bound','frozen joint source required')
    for rel,digest in m['files'].items():
        p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root),'source path escape');require(sha(root/p)==digest,'source drift '+rel)
    require({x for x in m['files'] if x.startswith('configs/experiment/')}=={f'configs/experiment/p1_{a}.yaml' for a in ARMS},'only two registered configs')
    assets(root)
    for arm in ARMS:validate_config(yaml.safe_load((root/f'configs/experiment/p1_{arm}.yaml').read_text(encoding='utf-8'))['experiment'],arm,assets(root))
    return m
def verify_remote_assets(root=ROOT):
    p=assets(root)
    for name,item in p['assets'].items():require(sha(item['path'])==item['sha256'],'actual input drift '+name)
    h=json.loads(Path(p['assets']['h1_receipt']['path']).read_text(encoding='utf-8'))
    require(h.get('complete') is True and h.get('checkpoint_sha256')==p['assets']['h1']['sha256'] and h.get('source_sha256')==p['source_h1_manifest_sha256'] and h.get('row_binding')==p['h1_row_binding'] and h.get('test_access') is False,'actual H1 producer receipt')
    from fresh_route_runtime import budget
    b=budget(root);e=b['capacity_evidence'];require(sha(e['path'])==e['sha256'],'actual capacity receipt drift')
    evidence=json.loads(Path(e['path']).read_text(encoding='utf-8'))
    require(evidence.get('data_capacity_increase_bytes')==b['capacity_increase_bytes'] and evidence.get('adjusted_initial_free_bytes')==b['adjusted_initial_free_bytes'] and evidence.get('original_initial_free_bytes')==b['initial_free_bytes'],'actual capacity accounting evidence')
    return p
