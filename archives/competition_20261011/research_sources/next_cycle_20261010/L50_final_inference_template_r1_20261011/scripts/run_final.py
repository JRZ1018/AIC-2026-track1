"""Bounded single test pass after actual FULL recovery; never auto retries."""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from final_contract import *

def process_start(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return fields[19]  # field22, after pid and comm

def stop_owned_child(child,identity):
    if child is None or child.poll() is not None:return
    # An unreaped Popen child retains its PID. A transient /proc read failure
    # cannot transfer this private session to another process.
    require(os.getpgid(child.pid)==child.pid,'owned child moved outside its private session')
    try:actual=process_start(child.pid)
    except OSError:actual=None
    if identity is not None and actual is not None:require(actual==identity,'refuse changed child start identity')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)

def build_command(args, audit, overlay, policy_path):
    command=[sys.executable,'-B','-u','scripts/infer_and_build_submission.py',
        '--experiment-config',audit['evidence']['config']['path'],
        '--checkpoint',audit['evidence']['final']['path'],'--expected-checkpoint-sha256',audit['final_sha256'],
        '--test-manifest',str(TEST_MANIFEST),'--output-dir',str(ROOT/'submissions/final'),
        '--source-git-commit',args.source_git_commit,'--batch-size',str(args.batch_size),'--workers',str(args.workers),
        '--four-view-scale384','--balanced-prior-uniform-soft','--balanced-prior-temperature-multiplier','0.5',
        '--balanced-prior-bias-strength','0.75','--balanced-prior-selection-report','inputs/p7_selection.json',
        '--expected-balanced-prior-report-sha256',P7_SHA,'--source-sha256',args.source_sha256,
        '--cpu-audit',str(args.cpu_audit.resolve()),'--cpu-audit-sha256',args.cpu_audit_sha256,
        '--inference-dataset-config',str(overlay),'--resource-policy',str(policy_path),
        '--resource-policy-sha256',sha(policy_path)]
    if args.save_features:command.append('--save-features')
    return command

def build_return_manifest(audit, package_hashes, source_sha):
    # Precise local-key inventory for root's downloader. No images are returned.
    files={}
    def add(key,path):
        path=Path(path);require(path.is_file() and key not in files,'missing/duplicate return asset')
        files[key]={'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size}
    for name in ('final','last','metrics','config','full_source','selection','launch_receipt','full_budget','h1','h1_receipt'):
        add('full/'+name+Path(audit['evidence'][name]['path']).suffix,audit['evidence'][name]['path'])
    recovery=read(audit['evidence']['full_budget']['path'])['disk_expansion_recovery']
    require(sha(recovery['path'])==recovery['sha256'],'actual capacity recovery return SHA differs')
    add('full/disk_expansion_recovery.json',recovery['path'])
    payload=read(FULL_ROOT/'inputs/full_assets.json')
    for name in ('clip','manifest','split','class_mapping'):
        add('official/'+name+Path(payload['assets'][name]['path']).suffix,payload['assets'][name]['path'])
    add('official/test_manifest.csv',TEST_MANIFEST)
    for rel in ('source_manifest.json','inputs/p7_selection.json','artifacts/full_final_cpu_audit.json',
                'artifacts/inference_freeze.json','artifacts/inference_dataset.yaml','artifacts/resource_policy.json',
                'artifacts/inference_command.json','artifacts/inference.log'):
        add('inference/'+rel,ROOT/rel)
    for name in ('pred_results.csv','pred_results.zip','submission_manifest.json','fused_logits.pt','inference_artifacts.json'):
        add('submission/'+name,ROOT/'submissions/final'/name)
    if (ROOT/'submissions/final/four_view_features.pt').exists():
        add('submission/four_view_features.pt',ROOT/'submissions/final/four_view_features.pt')
    for rt in audit['training_runtime_receipts']:
        add('full/runtime_receipts/'+Path(rt['path']).name,rt['path'])
    for rel in ('inputs/fixed_graph_protocol.json','inputs/actual_full_binding.json'):
        add('inference/'+rel,ROOT/rel)
    if audit['selected_route']=='L50_GRAPH':
        add('inference/artifacts/final_graph_budget.json',ROOT/'artifacts/final_graph_budget.json')
        add('inference/artifacts/final_graph_result.json',ROOT/'artifacts/final_graph_result.json')
    # Exact actual typed FULL implementation and separate inference source.
    for rel in verify_source()['files']:
        add('inference_source/'+rel,ROOT/rel)
    for rel in verify_full_source(FULL_ROOT)['files']:
        add('full_source/'+rel,FULL_ROOT/rel)
    return dict(schema='aictrack1.fresh_route.final_return_manifest.v1',complete=True,
        inference_source_sha256=source_sha,full_training_source_sha256=FULL_SOURCE_SHA,
        final_sha256=audit['final_sha256'],last_sha256=audit['last_sha256'],selected_arm=audit['selected_arm'],selected_route=audit['selected_route'],
        package_hashes=package_hashes,files=files,test_images_included=False,
        test_used_for_training_or_parameter_updates=False,checkpoint_selection='fixed full20 final',
        postprocess_selection='pre-frozen validation P7; no test labels or online scores')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha256',required=True);p.add_argument('--cpu-audit',required=True,type=Path)
    p.add_argument('--cpu-audit-sha256',required=True);p.add_argument('--source-git-commit',required=True)
    p.add_argument('--deadline-unix',required=True,type=float);p.add_argument('--initial-free-bytes',required=True,type=int)
    p.add_argument('--disk-max-bytes',type=int,required=True);p.add_argument('--min-free-bytes',type=int,default=3*GIB)
    p.add_argument('--min-available-memory-bytes',type=int,default=2*GIB)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--save-features',action='store_true');p.add_argument('--final-graph-budget-seconds',type=float,default=7200.);args=p.parse_args()
    require(sys.platform=='linux' and ROOT.resolve()==REMOTE_ROOT.resolve(), 'root-owned independent Linux inference ROOT required')
    require(args.cpu_audit.resolve()==ROOT/'artifacts/full_final_cpu_audit.json','canonical CPU audit path required')
    require(1<=args.batch_size<=64 and 0<=args.workers<=8,'unbounded inference batching is forbidden')
    verify_source(expected=args.source_sha256);verify_full_source(FULL_ROOT)
    require(sha(args.cpu_audit)==args.cpu_audit_sha256,'actual CPU recovery SHA differs');audit=read(args.cpu_audit)
    require(audit.get('complete') is True and audit.get('full_training_source_sha256')==FULL_SOURCE_SHA
        and audit.get('inference_source_sha256')==args.source_sha256 and audit.get('test_access') is False,'actual joint FULL CPU recovery required')
    policy={name:getattr(args,name) for name in ('deadline_unix','initial_free_bytes','disk_max_bytes','min_free_bytes','min_available_memory_bytes')}
    launch=read(audit['evidence']['launch_receipt']['path'])
    full_budget=read(audit['evidence']['full_budget']['path'])
    policy=bind_inference_capacity_policy(policy,full_budget,started_at_unix=time.time())
    resource_check(ROOT,policy,reserve=GIB)
    require(not LOCK.exists(),'canonical GPU queue is still held; this launcher does not wait or retry')
    require(sha(ROOT/'inputs/p7_selection.json')==P7_SHA,'frozen P7 changed')
    for name,item in audit['evidence'].items():require(sha(item['path'])==item['sha256'],'CPU audited artifact changed: '+name)
    output=ROOT/'submissions/final';require(not output.exists(),'no-overwrite final output already exists')
    policy_path=ROOT/'artifacts/resource_policy.json';write_new(policy_path,policy)
    import yaml
    config=Path(audit['evidence']['config']['path']);raw=yaml.safe_load(config.read_text())['experiment']
    original=yaml.safe_load((config.parent/raw['dataset_config']).read_text())
    import copy
    dataset=copy.deepcopy(original);dataset['dataset']['sources']['test']={'root':'test','label_source':'none'}
    dataset['dataset']['dataset_root']=OFFICIAL_DATA_ROOT;validate_dataset_override(original,dataset)
    overlay=ROOT/'artifacts/inference_dataset.yaml'
    with overlay.open('x',encoding='utf-8') as handle:yaml.safe_dump(dataset,handle,sort_keys=False)
    frozen=dict(schema='aictrack1.fresh_route.final_inference_freeze.v1',complete=True,actual_cpu_audit_sha256=args.cpu_audit_sha256,
        final_sha256=audit['final_sha256'],last_sha256=audit['last_sha256'],selected_arm=audit['selected_arm'],selected_route=audit['selected_route'],
        inference_source_sha256=args.source_sha256,full_training_source_sha256=FULL_SOURCE_SHA,
        classifier_head_source=audit['classifier_head_source'],view_protocol=VIEW,p7=dict(temperature=.5,strength=.75,iterations=60),
        p7_selection_sha256=P7_SHA,test_manifest_expected_sha256=TEST_SHA,feature_export_enabled=args.save_features,
        test_access=False,model_parameter_updates=False,frozen_epoch=time.time())
    write_new(ROOT/'artifacts/inference_freeze.json',frozen)
    if audit['selected_route']=='L50_GRAPH':
        require(0<args.final_graph_budget_seconds<=7200.,'explicit bounded independent final graph budget required')
        write_new(ROOT/'artifacts/final_graph_budget.json',dict(schema='aictrack1.fresh_route.final_graph.budget.v1',source_manifest_sha256=args.source_sha256,checkpoint_sha256=audit['final_sha256'],total_wall_seconds=args.final_graph_budget_seconds,deadline_unix=args.deadline_unix,original_confirmation_ledger_untouched=True,parameters=read(ROOT/'inputs/fixed_graph_protocol.json')))
    command=build_command(args,audit,overlay,policy_path);write_new(ROOT/'artifacts/inference_command.json',{'command':command})
    child=None;identity=None;token=None
    result=dict(schema='aictrack1.fresh_route.final_controller_result.v1',complete=False,test_access=False,no_automatic_retry=True)
    try:
        with (ROOT/'artifacts/inference.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),stdin=subprocess.DEVNULL,start_new_session=True)
            identity=process_start(child.pid)
            write_new(ROOT/'artifacts/inference_process.json',{'pid':child.pid,'starttime':identity,'started_epoch':time.time()})
            from aictrack1.engine.training_queue import read_training_lock
            while child.poll() is None:
                resource_check(ROOT,policy);require(process_start(child.pid)==identity,'owned child identity changed')
                record=read_training_lock(LOCK)
                if record is not None:
                    require(record.queue_pid==child.pid and record.owner=='stage2-final-single-checkpoint-inference'
                        and (token is None or token==record.token),'unexpected canonical lease owner')
                    token=record.token;result['test_access']=True
                time.sleep(2)
            require(child.returncode==0 and not LOCK.exists(),'final child failed or owned lease was retained')
        from aictrack1.data.class_mapping import ClassMapping
        rows=read_test_rows(TEST_MANIFEST);names=validate_unlabeled_test_rows(rows)
        mapping=ClassMapping.from_csv(read(FULL_ROOT/'inputs/full_assets.json')['assets']['class_mapping']['path'])
        hashes=audit_package(output,names,mapping,audit)
        inventory=build_return_manifest(audit,hashes,args.source_sha256)
        write_new(ROOT/'artifacts/return_manifest.json',inventory)
        result.update(complete=True,test_access=True,status='single_model_full20_P7_csv_zip_verified_local_restore_pending',
            hashes=hashes,final_sha256=audit['final_sha256'],return_manifest_sha256=sha(ROOT/'artifacts/return_manifest.json'))
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        if child is not None and child.poll() is None:
            try:
                stop_owned_child(child,identity)
            except ProcessLookupError:pass
            # Never remove any lock after a forced interruption. Root inspects its token.
        result['finished_epoch']=time.time();write_new(ROOT/'artifacts/controller_result.json',result)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':
    def stop(*_):raise RuntimeError('owned final inference interrupted; preserve evidence/lock')
    signal.signal(signal.SIGTERM,stop)
    raise SystemExit(main())
