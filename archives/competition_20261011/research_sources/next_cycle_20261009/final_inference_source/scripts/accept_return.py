"""SHA, submission and actual joint-model CPU recovery of a downloaded result."""
from __future__ import annotations
import argparse,csv,json,os,random,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from final_contract import *

def verify_return_files(folder,manifest):
    folder=Path(folder).resolve();require(manifest.get('schema')=='aictrack1.p1.final_return_manifest.v1' and manifest.get('complete') is True,'complete final return inventory required')
    require(manifest.get('full_training_source_sha256')==FULL_SOURCE_SHA and manifest.get('test_images_included') is False,'wrong full source/image return scope')
    paths={}
    for key,item in manifest['files'].items():
        rel=Path(key);target=folder/rel
        require(not rel.is_absolute() and '..' not in rel.parts and not target.is_symlink() and target.resolve().is_relative_to(folder),'unsafe local inventory path')
        require(target.is_file() and target.stat().st_size==item['bytes'] and sha(target)==item['sha256'],'downloaded asset drift: '+key)
        paths[key]=target
    require(manifest['files']['full/final.pt']['sha256']==manifest['final_sha256'] and manifest['files']['full/last.pt']['sha256']==manifest['last_sha256'],'inventory selected model differs')
    return paths

def accept(folder,manifest,*,model_recovery=False):
    paths=verify_return_files(folder,manifest)
    from aictrack1.data.class_mapping import ClassMapping
    mapping=ClassMapping.from_csv(paths['official/class_mapping.csv'])
    rows=read_test_rows(paths['official/test_manifest.csv']);names=validate_unlabeled_test_rows(rows)
    cpu=read(paths['inference/artifacts/full_final_cpu_audit.json'])
    require(cpu.get('schema')=='aictrack1.p1.full_final_cpu_audit.v1' and cpu.get('complete') is True
        and cpu['final_sha256']==manifest['final_sha256'] and cpu['last_sha256']==manifest['last_sha256']
        and cpu['full_training_source_sha256']==FULL_SOURCE_SHA,'downloaded CPU audit/model differs')
    hashes=audit_package(Path(folder)/'submission',names,mapping,cpu)
    source=read(paths['inference/source_manifest.json'])
    require(sha(paths['inference/source_manifest.json'])==manifest['inference_source_sha256'],'downloaded inference source differs')
    for rel,digest in source['files'].items():require(sha(paths['inference_source/'+rel])==digest,'downloaded inference implementation differs')
    full=read(paths['full/full_source.json']);require(sha(paths['full/full_source.json'])==FULL_SOURCE_SHA,'downloaded full source differs')
    for rel,digest in full['files'].items():require(sha(paths['full_source/'+rel])==digest,'downloaded trained implementation differs')
    result=dict(schema='aictrack1.p1.local_final_acceptance.v1',complete=True,sha_files=len(paths),
        csv_zip_verified=True,package_hashes=hashes,final_sha256=manifest['final_sha256'],last_sha256=manifest['last_sha256'],
        actual_model_cpu_recovery=False,test_image_reads=0,gpu_access=False)
    if model_recovery:
        import torch,yaml
        runtime(torch);require(not torch.cuda.is_available(),'local restore must hide CUDA')
        from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
        from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
        from aictrack1.models.peft import configure_visual_peft
        from aictrack1.models.heads import load_classifier_head_checkpoint
        from aictrack1.methods.confidence import ClasswiseHistoryWeighting
        from aictrack1.engine.provenance import CheckpointProvenance
        last=torch.load(paths['full/last.pt'],map_location='cpu',weights_only=True,mmap=True)
        final=torch.load(paths['full/final.pt'],map_location='cpu',weights_only=True,mmap=True)
        metrics=read(paths['full/metrics.json']);raw=yaml.safe_load(paths['full/config.yaml'].read_text())['experiment']
        validate_metrics(metrics,last,final,config_sha=sha(paths['full/config.yaml']),final_sha=manifest['final_sha256'],last_sha=manifest['last_sha256'],head_sha=sha(paths['full/h1.pt']),arm=manifest['selected_arm'])
        parameters=validate_joint_tensors(last,final,torch)
        require(sha(paths['official/clip.pt'])==CLIP_SHA and sha(paths['official/manifest.csv'])==MANIFEST_SHA and mapping.digest()==MAPPING_SHA,'downloaded official backbone/mapping/manifest differs')
        clip,_,_=load_clip_vit_b32(paths['official/clip.pt'],CLIP_SHA,torch.device('cpu'));visual=clip.visual.float();del clip
        enable_interpolated_visual_resolution(visual,target_resolution=336);configure_visual_peft(visual,'full_visual')
        head,h1=load_classifier_head_checkpoint(paths['full/h1.pt'],feature_dim=512,num_classes=750,
            expected_clip_weight_sha256=CLIP_SHA,expected_class_mapping_sha256=MAPPING_SHA,
            expected_split_sha256=sha(paths['official/split.csv']),
            expected_provenance=CheckpointProvenance('stage2',MAPPING_SHA,MANIFEST_SHA,sha(paths['official/split.csv']),CLIP_SHA,750))
        # Pin the complete full-H1 state and genuine row binding, without image I/O.
        require(h1['epoch']==100 and h1['training_scope']=='full_train_fixed_epochs' and h1['row_binding']['counts']=={'train':148695,'validation':0}
            and read(paths['full/h1_receipt.json'])['checkpoint_sha256']==sha(paths['full/h1.pt']),'downloaded full H1100 binding differs')
        model,delta=restore_joint(visual,head,parameters,torch);opt,sched,scaler=restore_optimizer(model,last,torch,raw['training']['head_learning_rate'])
        with paths['official/manifest.csv'].open(encoding='utf-8-sig',newline='') as handle:train=list(csv.DictReader(handle))
        require(len(train)==148695 and len({r['sample_id'] for r in train})==148695,'official full train-ID coverage differs')
        labels=[int(r['internal_class_index']) for r in train];indices=list(range(len(train)))
        c=last['confidence_weighting'];require(c['sample_indices'].tolist()==indices and c['class_ids'].tolist()==labels and c['epochs_updated']==20,'downloaded AC20 full IDs/labels differ')
        ac=ClasswiseHistoryWeighting(torch.tensor(indices),torch.tensor(labels),momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75);ac.load_state_dict(c)
        for name,value in ac.state_dict().items():
            if isinstance(value,torch.Tensor):require(torch.equal(value,c[name]) and (not value.is_floating_point() or torch.isfinite(value).all().item()),'downloaded AC restore differs')
            else:require(value==c[name],'downloaded AC scalar differs')
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
        with torch.inference_mode():logits=model(torch.zeros(1,3,336,336))
        require(logits.shape==(1,750) and torch.isfinite(logits).all().item(),'downloaded model CPU forward failed')
        result.update(actual_model_cpu_recovery=True,joint154_restored=True,optimizer154_restored=True,
            full_ac20_restored=True,head_source='joint_full_final_state',head_initializer_max_abs_delta=delta,runtime=runtime(torch))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--return-root',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--expected-manifest-sha256',required=True);p.add_argument('--cpu-model-recovery',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();require(sha(args.manifest)==args.expected_manifest_sha256,'return manifest SHA differs')
    verify_source(expected=read(args.manifest)['inference_source_sha256'])
    result=accept(args.return_root,read(args.manifest),model_recovery=args.cpu_model_recovery)
    if args.output:write_new(args.output,result)
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
