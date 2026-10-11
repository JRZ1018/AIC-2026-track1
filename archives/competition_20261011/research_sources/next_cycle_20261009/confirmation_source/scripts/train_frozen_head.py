"""Fresh train-only H1100 on positively rebound official CLIP336 feature rows."""
from __future__ import annotations
import argparse,json,math,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from confirmation_contract import HEAD_CONFIG,SCOPE,SELECTION,entry,sha,write_new,load_rebound_cache,validate_h1,resource_check
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    deployment,assets=entry(a.config,a.resume)
    import numpy as np
    import torch
    from aictrack1.engine.frozen_head import FrozenHeadTrainer,FrozenHeadConfig
    from aictrack1.engine.provenance import CheckpointProvenance
    from aictrack1.models.heads import build_classifier_head
    cache,binding=load_rebound_cache(deployment,assets)
    train=[i for i,p in enumerate(cache.partitions) if p=='train'];query=[i for i,p in enumerate(cache.partitions) if p=='validation']
    x,y=cache.features[train].float(),cache.labels[train]
    if torch.bincount(y,minlength=750).min()<1:raise ValueError('H1 missing train class')
    random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008);torch.cuda.manual_seed_all(20261008)
    provenance=CheckpointProvenance('stage2',cache.metadata.class_mapping_sha256,cache.metadata.manifest_sha256,cache.metadata.split_sha256,cache.metadata.clip_weight_sha256,750)
    trainer=FrozenHeadTrainer(build_classifier_head('linear',512,750),FrozenHeadConfig(**HEAD_CONFIG),device='cuda',num_classes=750,provenance=provenance)
    out=ROOT/'artifacts/experiments/confirmation_H1'
    if a.resume is None:
        out.mkdir(parents=True,exist_ok=False)
        (out/'config.yaml').write_bytes(a.config.read_bytes())
        (out/'command.txt').write_text(json.dumps(sys.argv)+'\n',encoding='utf-8')
    else:
        state=torch.load(a.resume,map_location='cpu',weights_only=True)
        if state.get('row_binding')!=binding or state.get('config_sha256')!=sha(a.config) or state.get('source_sha256')!=sha(ROOT/'source_manifest.json') or not 0<state['epoch']<=100:raise ValueError('H1 exact resume row/config/source/epoch differs')
        trainer.load_state_dict(state)
    def state_now():
        state=trainer.state_dict();state.update(training_scope=SCOPE,head={'kind':'linear'},row_binding=binding,selection_rule=SELECTION,
            feature_cache_metadata=cache.metadata.__dict__|{'training_scope':SCOPE,'training_record_count':len(train)},
            original_feature_identity_sha256=sha(ROOT/'inputs/feature_identity.json'),fresh_feature_extraction=False,
            config_sha256=sha(a.config),source_sha256=sha(ROOT/'source_manifest.json'),git_commit='unknown_archive_source_bytes_frozen')
        return state
    while trainer.epoch<100:
        resource_check();loss=trainer.train_epoch(x,y)
        if not math.isfinite(loss):raise RuntimeError('nonfinite H1 train loss')
        trainer.history.append(dict(epoch=trainer.epoch,train_loss=loss,learning_rate=float(trainer.optimizer.param_groups[0]['lr'])))
        print(json.dumps({'phase':'H1','epoch':trainer.epoch,'train_loss':loss}),flush=True)
        state=state_now();resource_check(20*1024**2);torch.save(state,out/'last.pt.tmp');(out/'last.pt.tmp').replace(out/'last.pt')
    loaded=torch.load(out/'last.pt',map_location='cpu',weights_only=True);validate_h1(loaded,assets,binding)
    restored=FrozenHeadTrainer(build_classifier_head('linear',512,750),FrozenHeadConfig(**HEAD_CONFIG),device='cpu',num_classes=750,provenance=provenance)
    restored.load_state_dict(loaded)
    if restored.epoch!=100 or restored.scheduler.last_epoch!=100 or len(restored.optimizer.state)!=2 or any(not torch.equal(loaded['model'][k],v) for k,v in restored.model.state_dict().items()):raise RuntimeError('actual H1 CPU restoration failed')
    for slot in restored.optimizer.state.values():
        if any(not torch.isfinite(v).all() for v in slot.values() if isinstance(v,torch.Tensor)):raise RuntimeError('nonfinite H1 optimizer')
    # No query loss/metric was evaluated during fitting or checkpoint selection.
    parity=restored.evaluate(cache.features[query].float(),cache.labels[query])
    receipt=dict(complete=True,checkpoint_sha256=sha(out/'last.pt'),epoch=100,training_seed=20261008,source_sha256=sha(ROOT/'source_manifest.json'),
        row_binding=binding,original_feature_identity_sha256=sha(ROOT/'inputs/feature_identity.json'),fresh_feature_extraction=False,
        actual_cpu_optimizer_scheduler_scaler_model_rng_restored=True,cuda_rng_serialized=len(loaded['cuda_rng_state_all'])>0,
        terminal_cache_parity_top1=parity['top1'],validation_selection=False,test_access=False,scope=SCOPE,
        train_count=len(train),validation_count=len(query),split_sha256=cache.metadata.split_sha256,
        runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_num_threads':torch.get_num_threads()})
    write_new(out/'h1_receipt.json',receipt)
    # Queue completion receipt; fixed final and no best-model selection.
    write_new(out/'metrics.json',dict(schema='aictrack1.confirmation.h1.metrics.v1',config_sha256=sha(a.config),run_id='20261009-confirmation-H1',epochs_completed=100,selected_checkpoint='last.pt',last_checkpoint_sha256=sha(out/'last.pt'),h1_receipt_sha256=sha(out/'h1_receipt.json'),test_access=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
