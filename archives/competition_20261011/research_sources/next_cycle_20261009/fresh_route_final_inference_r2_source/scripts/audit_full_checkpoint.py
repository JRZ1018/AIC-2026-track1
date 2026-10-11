"""CPU-only actual FULL20 recovery before permitting final test inference."""
from __future__ import annotations
import argparse, csv, importlib, json, os, random, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from final_contract import *

def dependency_check(full):
    verify_full_source(full)
    import torch, numpy, yaml
    runtime(torch)
    from aictrack1.engine.peft import configure_classifier_trainability_from_state,load_trainable_state_dict
    from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
    from aictrack1.evaluation.four_view import fixed_four_view_logits
    from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias
    from aictrack1.submission import validate_submission_csv,validate_submission_zip
    import subprocess
    result=subprocess.run([sys.executable,'-B',str(Path(full)/'scripts/train_n1_peft.py'),'--help'],capture_output=True,text=True,check=True,env=os.environ.copy())
    require('--config' in result.stdout and '--resume' in result.stdout,'actual typed FULL trainer CLI import failed')
    return dict(complete=True,actual_full_source_sha256=FULL_SOURCE_SHA,actual_full_trainer_cli=True,
                runtime=runtime(torch),gpu_access=False,test_access=False,model_loaded=False)

def audit(args):
    import numpy as np, torch, yaml
    require(not torch.cuda.is_available(),'audit must hide CUDA')
    runtime_value=runtime(torch)
    verify_source(expected=args.source_sha256);full=args.full_root.resolve()
    require(full==FULL_ROOT.resolve() and sys.platform=='linux','actual audit uses the independent remote FULL root')
    verify_full_source(full)
    # These are the true, unchanged677 modules, not our audit stubs.
    sys.path.insert(0,str(full/'scripts'))
    contract=importlib.import_module('p1_contract')
    reference=importlib.import_module('full_h1_reference')
    payload=contract.verify_remote_assets(full)
    choice=contract.selection(args.selection,args.selection_sha256)
    contract.verify_selection_evidence(choice)
    arm=choice['selected_arm'];require(arm==args.arm,'FULL choice/arm differs')
    require(payload['fresh_confirmation_receipt_sha256']==args.selection_sha256 and payload['selected_arm']==arm and payload['selected_route']==choice['selected_route'],'actual FULL assets/selection differs')
    launch=read(args.launch_receipt)
    require(launch.get('schema')=='aictrack1.fresh_route.full.launch_result.v1' and launch.get('arm')==arm and launch.get('return_code')==0
        and launch.get('synthetic_smoke') is False and launch.get('source_manifest_sha256')==FULL_SOURCE_SHA
        and launch.get('fresh_confirmation_receipt_sha256')==args.selection_sha256 and launch.get('selected_route')==choice['selected_route'],
        'successful actual canonical FULL launch receipt required')
    binding=read(ROOT/'inputs/actual_full_binding.json')
    require(binding['full_training_source_sha256']==FULL_SOURCE_SHA and binding['confirmation_sha256']==args.selection_sha256 and binding['selected_route']==choice['selected_route'], 'actual binding/confirmation differs')
    config=full/f'configs/experiment/p1_full_{arm}.yaml';raw=yaml.safe_load(config.read_text())['experiment']
    contract.validate_config(raw,arm,payload)
    out=full/f'artifacts/experiments/FULL_{arm}'
    require(out.resolve()==(config.parent/raw['output_dir']).resolve(),'FULL output/config path differs')
    require(not (out/'p1_screen_receipt.json').exists(),'planned partial screen cannot be final')
    last_path=out/'last.pt';final_path=out/'final.pt';metrics_path=out/'metrics.json'
    last_sha=sha(last_path);final_sha=sha(final_path);config_sha=sha(config);metrics=read(metrics_path)
    last=torch.load(last_path,map_location='cpu',weights_only=True,mmap=True)
    final=torch.load(final_path,map_location='cpu',weights_only=True,mmap=True)
    h1_sha=payload['assets']['h1']['sha256']
    require(h1_sha==FULL_H1_SHA,'legal full H1 exact identity differs')
    require(last_sha==binding['last_sha256'] and final_sha==binding['final_sha256'],'actual checkpoint changed since binding')
    actual_runtime=[]
    for rp in sorted((full/'artifacts/runtime_receipts').glob(arm+'_*.json')):
        rv=read(rp)
        require(rv.get('schema')=='aictrack1.fresh_route.full.runtime.v1' and rv.get('arm')==arm and rv.get('selected_route')==choice['selected_route'] and rv.get('source_manifest_sha256')==FULL_SOURCE_SHA and rv.get('config_sha256')==config_sha and rv.get('environment')==RUNTIME and rv.get('actual_torch_threads')==4 and rv.get('test_access') is False and rv.get('full_budget_sha256')==sha(full/'inputs/full_budget.json') and rv.get('fresh_confirmation_receipt_sha256')==args.selection_sha256,'actual FULL training runtime differs')
        actual_runtime.append({'path':str(rp),'sha256':sha(rp),'receipt':rv})
    require(actual_runtime,'actual FULL training runtime missing')
    require(len({rv['receipt'].get('actual_torch_interop_threads') for rv in actual_runtime})==1,'FULL runtime interop drift across attempts')
    validate_metrics(metrics,last,final,config_sha=config_sha,final_sha=final_sha,last_sha=last_sha,head_sha=h1_sha,arm=arm)
    for value in (last,final,metrics):
        require(value.get('fresh_confirmation_receipt_sha256')==args.selection_sha256 and value.get('full_budget_sha256')==sha(full/'inputs/full_budget.json'),'checkpoint/metrics actual confirmation/budget differs')
    route=choice['selected_route']
    parameters=validate_joint_tensors(last,final,torch,route=route)
    for name,item in payload['assets'].items():
        if name in ('split','manifest'):
            field='split_sha256' if name=='split' else 'train_manifest_sha256'
            require(last[field]==final[field]==item['sha256'],'checkpoint full asset binding differs: '+name)
    from aictrack1.data.contracts import load_dataset_config
    from aictrack1.data.indexed_images import read_indexed_images
    from aictrack1.data.class_mapping import ClassMapping
    from aictrack1.engine.environment_identity import verified_environment_hash
    from aictrack1.engine.provenance import CheckpointProvenance
    from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
    from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
    from aictrack1.models.peft import configure_visual_peft
    from aictrack1.methods.confidence import ClasswiseHistoryWeighting
    data_path=(config.parent/raw['dataset_config']).resolve();data=load_dataset_config(data_path)
    rows=read_indexed_images(data,source='train',manifest_path=raw['manifest'],split_path=raw['split'])
    require(len(rows)==148695 and len({r.sample_id for r in rows})==148695,'full manifest row coverage differs')
    indices=[r.sample_index for r in rows];labels=[r.internal_class_index for r in rows]
    mapping=ClassMapping.from_csv(data.class_mapping);require(len(mapping)==750 and mapping.digest()==MAPPING_SHA,'official750 mapping differs')
    require(metrics.get('dataset_config_sha256')==sha(data_path) and metrics.get('manifest_sha256')==MANIFEST_SHA
        and metrics.get('split_sha256')==payload['assets']['split']['sha256'] and verified_environment_hash(metrics), 'metrics data/environment provenance differs')
    clip,_,metadata=load_clip_vit_b32(payload['assets']['clip']['path'],CLIP_SHA,torch.device('cpu'))
    visual=clip.visual.float();del clip
    enable_interpolated_visual_resolution(visual,target_resolution=336);configure_visual_peft(visual,'full_visual')
    head,h1=reference.load_new_fold_head(payload['assets']['h1']['path'],feature_dim=512,num_classes=750,
        expected_clip_weight_sha256=CLIP_SHA,expected_class_mapping_sha256=MAPPING_SHA,
        expected_split_sha256=payload['assets']['split']['sha256'],
        expected_provenance=CheckpointProvenance('stage2',MAPPING_SHA,MANIFEST_SHA,payload['assets']['split']['sha256'],CLIP_SHA,750))
    _,actual_binding=reference.bind_rows([r.sample_id for r in rows],labels,[r.partition for r in rows],rows)
    require(h1['row_binding']==actual_binding,'full H1 all148695 positive row binding differs')
    model,head_delta=restore_joint(visual,head,parameters,torch,route=route)
    optimizer,scheduler,scaler=restore_optimizer(model,last,torch,raw['training']['head_learning_rate'])
    acstate=last['confidence_weighting'];require(acstate['sample_indices'].tolist()==indices and acstate['class_ids'].tolist()==labels and acstate['epochs_updated']==20 and bool(acstate['seen'].all()),'full AC exact global-ID/label20 binding differs')
    ac=ClasswiseHistoryWeighting(torch.tensor(indices),torch.tensor(labels),momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75)
    ac.load_state_dict(acstate)
    for key,value in ac.state_dict().items():
        expected=acstate[key]
        if isinstance(value,torch.Tensor):require(torch.equal(value,expected) and (not value.is_floating_point() or torch.isfinite(value).all().item()),'AC state did not restore: '+key)
        else:require(value==expected,'AC scalar did not restore: '+key)
    random.setstate(last['python_rng_state']);require(random.getstate()==last['python_rng_state'],'Python RNG restore differs')
    n=last['numpy_rng_state'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    actual=np.random.get_state();require(actual[0]==n[0] and actual[1].tolist()==list(n[1]) and actual[2:]==tuple(n[2:]),'NumPy RNG restore differs')
    torch.random.set_rng_state(last['torch_rng_state']);require(torch.equal(torch.random.get_rng_state(),last['torch_rng_state']),'Torch CPU RNG restore differs')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    with torch.inference_mode():logits=model(torch.zeros(1,3,336,336))
    require(logits.shape==(1,750) and torch.isfinite(logits).all().item(),'CPU synthesized336 restore forward failed')
    # Rehash after recovery; no claim that nonexistent or changing files are frozen.
    require(sha(last_path)==last_sha and sha(final_path)==final_sha and sha(config)==config_sha,'FULL checkpoint changed during CPU recovery')
    evidence={name:{'path':str(path),'sha256':sha(path)} for name,path in dict(
        final=final_path,last=last_path,metrics=metrics_path,config=config,full_source=full/'source_manifest.json',
        selection=args.selection,launch_receipt=args.launch_receipt,full_budget=full/'inputs/full_budget.json',h1=Path(payload['assets']['h1']['path']),
        h1_receipt=Path(payload['assets']['h1_receipt']['path'])).items()}
    result=dict(schema='aictrack1.fresh_route.full_final_cpu_audit.v1',complete=True,status='complete_typed_full20_cpu_restore',
        full_root=str(full),selected_arm=arm,selected_route=route,epoch=20,train_count=148695,validation_count=0,ac_epochs=20,
        trainable_tensors=(152 if route=='S' else 154),trainable_numel=(87849216 if route=='S' else 88233966),final_sha256=final_sha,last_sha256=last_sha,
        config_sha256=config_sha,metrics_sha256=sha(metrics_path),head_initializer_sha256=h1_sha,
        classifier_head_source=('legal_full_H1_frozen' if route=='S' else 'joint_fixed_last_state'),classifier_parameters_restored_equal=True,
        head_initializer_max_abs_delta=head_delta,final_last_all_tensors_equal=True,optimizer_each_moment_equal=True,
        scheduler20_restored=True,scaler_restored=True,ac_exact_full_global_ids_labels=True,rng_cpu_restored=True,
        cuda_rng_serialized_only=True,actual_cpu_synthetic_forward=True,actual_real_image_forward=False,
        official_clip_sha256=CLIP_SHA,class_mapping_sha256=MAPPING_SHA,train_manifest_sha256=MANIFEST_SHA,
        split_sha256=payload['assets']['split']['sha256'],inference_source_sha256=args.source_sha256,
        full_training_source_sha256=FULL_SOURCE_SHA,training_threads_evidence='actual immutable full training runtime receipts',training_runtime_receipts=actual_runtime,
        runtime=runtime_value,evidence=evidence,gpu_access=False,test_access=False,completed_epoch=time.time())
    write_new(args.output,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--full-root',type=Path,default=FULL_ROOT);p.add_argument('--dependency-check',action='store_true')
    p.add_argument('--arm',choices=['S','J5']);p.add_argument('--source-sha256');p.add_argument('--selection',type=Path)
    p.add_argument('--selection-sha256');p.add_argument('--launch-receipt',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.dependency_check: result=dependency_check(args.full_root)
    else:
        for name in ('arm','source_sha256','selection','selection_sha256','launch_receipt','output'):
            if getattr(args,name) is None:p.error('--'+name.replace('_','-')+' is required for actual audit')
        require(args.output.resolve()==ROOT/'artifacts/full_final_cpu_audit.json','canonical independent CPU audit output required')
        result=audit(args)
    print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
