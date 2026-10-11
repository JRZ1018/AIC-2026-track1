"""After real H1 and independent old-fold selection: build separate fresh C/winner source."""
from __future__ import annotations
import argparse,importlib.util,json,os,shutil,sys
from pathlib import Path
from confirmation_contract import ROOT,sha,read_json,write_new,verify_source,deployment,validate_h1
def validate_selection(selection):
    if (selection.get('schema')!='aictrack1.p1.full_selection.v1' or selection.get('test_access') is not False
            or selection.get('selection_split_sha256')!='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'
            or selection.get('selection_rule')!='fixed_final20' or selection.get('selected_arm') not in {'J5','J20'}):raise ValueError('independent fixed-final old-fold winner receipt required')
    if (selection.get('status')!='selected_recipe_for_confirmation' or selection.get('training_complete') is not True
            or selection.get('used_test_or_online_scores') is not False
            or selection.get('decision_basis')!='current_stage_train_derived_validation'
            or selection.get('training_scope')!='fixed_split_validation' or selection.get('checkpoint_role')!='last'
            or selection.get('checkpoint_epoch')!=20 or selection.get('train_count')!=118958 or selection.get('validation_count')!=29737
            or selection.get('head_learning_rate')!={'J5':5e-6,'J20':2e-5}[selection['selected_arm']]):raise ValueError('verified completed matched selection policy required')
    candidates=selection.get('candidates',[])
    chosen=[x for x in candidates if x.get('arm')==selection['selected_arm']]
    if len(chosen)!=1 or chosen[0].get('epoch')!=20 or chosen[0].get('checkpoint_role')!='last':raise ValueError('selected candidate not actual fixedlast20')
    for key in ['selected_fold_checkpoint_sha256','selected_fold_config_sha256','diagnostic_source_manifest_sha256','selection_protocol_sha256']:
        v=selection.get(key)
        if not isinstance(v,str) or len(v)!=64 or set(v)-set('0123456789abcdef'):raise ValueError('winner SHA missing '+key)
    if chosen[0].get('checkpoint_sha256')!=selection['selected_fold_checkpoint_sha256']:raise ValueError('winner checkpoint differs candidate')
    return selection['selected_arm']
def create_visual_source(target,payload,winner,selection_receipt_path,selection_receipt_sha256,h1_source_sha256):
    if target.exists():raise FileExistsError('immutable confirmation visual source already prepared; verify instead of overwrite')
    original=read_json(ROOT/'clone_receipt.json')
    for rel in original['source_files']:
        path=ROOT/rel;out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out)
    (target/'inputs/selection_receipt.json').write_bytes(selection_receipt_path.read_bytes())
    (target/'inputs/confirmation_asset_receipt.json').write_bytes((ROOT/'inputs/confirmation_assets.json').read_bytes())
    contract=target/'scripts/p1_contract.py';text=contract.read_text();text=text.replace('ARMS = {"C": 0.0, "J5": 5e-6, "J20": 2e-5}',f'ARMS = {{"C": 0.0, "{winner}": {5e-6 if winner=="J5" else 2e-5!r}}}')
    contract.write_text(text,encoding='utf-8')
    sys.path.insert(0,str(target/'scripts'))
    for name in ['p1_contract','prepare_p1']:sys.modules.pop(name,None)
    import prepare_p1
    prepare_p1.prepare(payload,root=target)
    # Both independent fresh runs inherit identical runtime policy before torch import.
    runner=target/'scripts/run_p1.py';text=runner.read_text();anchor='from p1_contract import ARMS,assets,sha,verify_files,verify_remote_assets'
    text=text.replace(anchor,"os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')\n"+anchor)
    runner.write_text(text,encoding='utf-8')
    guard=target/'scripts/mainline_guard.py';text=guard.read_text();anchor='    manifest=verify_files();payload=verify_remote_assets()'
    text=text.replace(anchor,"    if ROOT.resolve()!=Path('/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009/prepared_visual'):raise RuntimeError('confirmation exact data-disk visual ROOT required')\n    import torch\n    if any(os.environ.get(k)!=v for k,v in {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items()):raise RuntimeError('confirmation runtime drift')\n    torch.set_num_threads(4)\n    if torch.get_num_threads()!=4:raise RuntimeError('confirmation actual threads drift')\n"+anchor)
    runtime_anchor='    validate_config(raw,arm,payload)'
    text=text.replace(runtime_anchor,runtime_anchor+"\n    runtime_path=ROOT/'artifacts/runtime_receipts'/f'{arm}.json'\n    runtime_path.parent.mkdir(parents=True,exist_ok=True)\n    runtime_path.write_text(json.dumps({'schema':'aictrack1.confirmation.runtime.v1','arm':arm,'source_sha256':sha(ROOT/'source_manifest.json'),'config_sha256':sha(config),'split_sha256':payload['assets']['split']['sha256'],'h1_sha256':payload['assets']['h1']['sha256'],'runtime':{'OMP_NUM_THREADS':os.environ['OMP_NUM_THREADS'],'MKL_NUM_THREADS':os.environ['MKL_NUM_THREADS'],'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads()},'test_access':False},indent=2)+'\\n',encoding='utf-8')")
    guard.write_text(text,encoding='utf-8')
    loader=target/'scripts/mainline_h1_contract.py';text=loader.read_text();text=text.replace("    if binding.get(\"counts\")", "    if binding != payload['h1_row_binding']:raise ValueError('confirmation exact H1 row binding differs')\n    if binding.get(\"counts\")")
    loader.write_text(text,encoding='utf-8')
    pair=target/'scripts/run_confirmation_pair.py'
    pair.write_text('''"""Fresh C and one independently selected J; original serial queue owns each lease."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from p1_contract import ROOT,ARMS,assets,verify_files
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume-arms',nargs='*',choices=list(ARMS),default=[])
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--initial-free-bytes',type=int,required=True)
    p.add_argument('--disk-max-bytes',type=int,default=20*1024**3);p.add_argument('--min-free-bytes',type=int,default=3*1024**3)
    a=p.parse_args();verify_files();payload=assets()
    if set(ARMS)!={'C',payload['selected_arm']}:raise ValueError('exact fresh C and single selected J required')
    results=[]
    for arm in ['C',payload['selected_arm']]:
        command=[sys.executable,'scripts/run_p1.py','--arm',arm,'--deadline-unix',str(a.deadline_unix),'--initial-free-bytes',str(a.initial_free_bytes),'--disk-max-bytes',str(a.disk_max_bytes),'--min-free-bytes',str(a.min_free_bytes)]
        if arm in a.resume_arms:command.append('--resume')
        code=subprocess.call(command,cwd=ROOT);results.append(dict(arm=arm,return_code=code))
        path=ROOT/'artifacts/queues/confirmation_pair_result.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(schema='aictrack1.confirmation.pair.v1',selected_arm=payload['selected_arm'],fresh_control=True,results=results,epoch=time.time()),indent=2)+'\\n')
        if code:return code
    return 0
if __name__=='__main__':raise SystemExit(main())
''',encoding='utf-8')
    # Remove the inactive arm config from this new package only; no exploratory source changes.
    for arm in {'J5','J20'}-{winner}:(target/f'configs/experiment/p1_{arm}.yaml').unlink(missing_ok=True)
    source=read_json(target/'source_manifest.json');source['files']={rel:sha(target/rel) for rel in source['files'] if (target/rel).is_file()}
    source['files']['scripts/run_confirmation_pair.py']=sha(pair)
    source.update(confirmation=True,selected_arm=winner,source_h1_manifest_sha256=h1_source_sha256,selection_receipt_sha256=selection_receipt_sha256,
        runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_num_threads':4},fresh_control=True)
    (target/'source_manifest.json').write_text(json.dumps(source,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='bound_fresh_confirmation_C_and_one_winner',selected_arm=winner,source_sha256=sha(target/'source_manifest.json'),root=str(target))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection-receipt',type=Path,required=True);p.add_argument('--selection-receipt-sha256',required=True);a=p.parse_args()
    if os.name=='nt':raise RuntimeError('must bind real H1 returned by root-owned Linux run')
    verify_source();d,assets=deployment()
    if sha(a.selection_receipt)!=a.selection_receipt_sha256:raise ValueError('selection receipt SHA drift')
    selection=read_json(a.selection_receipt);winner=validate_selection(selection)
    if selection.get('training_source_manifest_sha256')!=read_json(ROOT/'clone_receipt.json')['source_manifest_sha256']:raise ValueError('selection not from this frozen exploratory training source')
    # Receipt aliases alone are insufficient: rehash the actual selected evidence.
    for name in ['checkpoint','config','identity','paired_report','training_source']:
        item=selection.get('evidence',{}).get(name,{})
        if not Path(item.get('path','')).is_absolute() or sha(item['path'])!=item.get('sha256'):raise ValueError('actual winner evidence drift '+name)
    if selection['evidence']['checkpoint']['sha256']!=selection['selected_fold_checkpoint_sha256'] or selection['evidence']['config']['sha256']!=selection['selected_fold_config_sha256']:raise ValueError('selection alias/evidence mismatch')
    if selection['evidence']['training_source']['sha256']!=selection['training_source_manifest_sha256']:raise ValueError('selected source alias/evidence mismatch')
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    h1=ROOT/'artifacts/experiments/confirmation_H1/last.pt';receipt_path=h1.parent/'h1_receipt.json';receipt=read_json(receipt_path)
    state=torch.load(h1,map_location='cpu',weights_only=True,mmap=True);validate_h1(state,assets,assets['h1_row_binding'])
    if (receipt.get('complete') is not True or receipt.get('checkpoint_sha256')!=sha(h1) or receipt.get('row_binding')!=assets['h1_row_binding'] or receipt.get('validation_selection') is not False or receipt.get('test_access') is not False or receipt.get('source_sha256')!=sha(ROOT/'source_manifest.json')):raise ValueError('actual train-only H1 receipt invalid')
    payload=dict(schema='aictrack1.p1.assets.v1',status='verified_fold_assets',competition_stage='stage2',split_seed=20261009,training_seed=20261008,
        train_count=assets['counts']['train'],validation_count=assets['counts']['validation'],num_classes=750,dataset_root=d['dataset_root'],canonical_lock=d['canonical_lock'],
        h1_epoch=100,h1_training_count=assets['counts']['train'],h1_training_scope='fold_train_fixed_final_epochs',h1_validation_used_for_selection=False,
        h1_row_binding=assets['h1_row_binding'],terminal_cache_parity_top1=receipt['terminal_cache_parity_top1'],assets=d['assets']|{'h1':dict(path=str(h1),sha256=sha(h1)),'h1_receipt':dict(path=str(receipt_path),sha256=sha(receipt_path))},
        control=dict(role='fresh_official_clip_new_fold_h1_fixed20_no_checkpoint_resume'),confirmation=True,
        selection_receipt_sha256=a.selection_receipt_sha256,selected_arm=winner,source_h1_manifest_sha256=sha(ROOT/'source_manifest.json'),test_access=False)
    create_visual_source(ROOT/'prepared_visual',payload,winner,a.selection_receipt,a.selection_receipt_sha256,sha(ROOT/'source_manifest.json'))
if __name__=='__main__':main()
