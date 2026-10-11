"""Synthetic CUDA gradients and real save/load resume; never exports a candidate."""
import copy,json,random,shutil,os
from pathlib import Path
import numpy as np
import torch
from p1_contract import ROOT,assets,sha,MAPPING_SHA

def run(raw,config_path):
    from mainline_guard import reserve_save
    from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
    from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
    from aictrack1.models.peft import configure_visual_peft
    from aictrack1.models.token_attention_classifier import configure_readout
    from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict
    from mainline_h1_contract import load_new_fold_head
    from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
    payload=assets();torch.manual_seed(raw['training']['seed']);random.seed(20261008);np.random.seed(20261008)
    if not torch.cuda.is_available():raise RuntimeError('this entry is a root-authorized CUDA smoke')
    clip,_,meta=load_clip_vit_b32(payload['assets']['clip']['path'],payload['assets']['clip']['sha256'],'cuda')
    visual=clip.visual.float();del clip
    enable_interpolated_visual_resolution(visual,target_resolution=336)
    summary=configure_visual_peft(visual,'full_visual',lora_blocks=8)
    head,_=load_new_fold_head(payload['assets']['h1']['path'],feature_dim=512,num_classes=750,
        expected_clip_weight_sha256=meta.weight_sha256,expected_class_mapping_sha256=MAPPING_SHA,
        expected_split_sha256=payload['assets']['split']['sha256'])
    model=configure_readout(visual,head,normalize_features=True,setting=raw['token_readout'],device='cuda')
    lr=raw['training']['head_learning_rate']
    for parameter in head.parameters():parameter.requires_grad_(lr>0)
    params=[dict(params=list(visual.parameters()),lr=raw['training']['visual_learning_rate'])]
    if lr>0:params.append(dict(params=list(head.parameters()),lr=lr))
    opt=torch.optim.AdamW(params,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
    scaler=torch.amp.GradScaler('cuda');model.train()
    images=torch.randn(2,3,336,336,device='cuda');targets=torch.tensor([0,749],device='cuda')
    gradients={};overflow_skips=[];successful_updates=[]
    def step():
        for attempt in range(12):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):logits=model(images)
            per=_base_loss_per_sample(logits.float(),targets,kind='sce',q=.7,label_smoothing=0.,sce_alpha=raw['base_loss']['alpha'],sce_beta=raw['base_loss']['beta'],sce_log_zero=raw['base_loss']['log_zero'])
            loss=_weighted_base_loss(per,torch.tensor([1.,.02],device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite synthetic loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            active=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
            for name,parameter in active:
                if parameter.grad is None:raise RuntimeError('missing gradient '+name)
            if any(not torch.isfinite(p.grad).all() for _,p in active):
                previous=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()>=previous:raise RuntimeError('nonfinite gradients without scaler backoff')
                overflow_skips.append(dict(from_scale=previous,to_scale=scaler.get_scale()))
                continue # Known normal initial AMP overflow; no optimizer/scheduler step.
            for name,p in head.named_parameters():
                norm=0. if p.grad is None else float(p.grad.norm());gradients[name]=norm
                if lr>0 and norm<=0:raise RuntimeError('joint head has no actual gradient')
                if lr==0 and p.grad is not None:raise RuntimeError('C head is not frozen')
            torch.nn.utils.clip_grad_norm_([p for _,p in active],1.)
            scaler.step(opt);scaler.update();sched.step()
            successful_updates.append(dict(loss=float(loss.detach()),scale=scaler.get_scale(),scheduler_epoch=sched.last_epoch))
            return float(loss.detach())
        raise RuntimeError('12 AMP backoffs without a finite successful synthetic step')
    first=step();folder=ROOT/os.environ['P1_SMOKE_DIR'];folder.mkdir(parents=True,exist_ok=False)
    rng=np.random.get_state();rng=(rng[0],rng[1].tolist(),*rng[2:])
    checkpoint=dict(trainable_state=trainable_state_dict(model),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
        scaler=scaler.state_dict(),python_rng_state=random.getstate(),numpy_rng_state=rng,
        torch_rng_state=torch.random.get_rng_state(),cuda_rng_state_all=torch.cuda.get_rng_state_all(),config_sha256=sha(config_path))
    reserve_save(checkpoint);torch.save(checkpoint,folder/'resume.pt');del checkpoint
    uninterrupted_loss=step();expected=trainable_state_dict(model)
    state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    load_trainable_state_dict(model,state['trainable_state']);opt.load_state_dict(state['optimizer'])
    sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng_state']);np.random.set_state(state['numpy_rng_state']);torch.random.set_rng_state(state['torch_rng_state'])
    torch.cuda.set_rng_state_all(state['cuda_rng_state_all']);del state
    resumed_loss=step();actual=trainable_state_dict(model)
    maximum=max(float((actual[k]-v).abs().max()) for k,v in expected.items())
    if maximum>1e-7 or abs(uninterrupted_loss-resumed_loss)>1e-6:raise RuntimeError('next-step resume differs')
    expected_tensors=152 if lr==0 else 154;expected_count=87849216 if lr==0 else 88233966
    if len(actual)!=expected_tensors or sum(v.numel() for v in actual.values())!=expected_count:raise RuntimeError('unexpected fullvisual/head parameter count')
    receipt=dict(schema='aictrack1.fresh_route.cuda_smoke.v1',status='passed',synthetic_only=True,candidate=False,
        config_sha256=sha(config_path),full_visual_tensors=len(summary.trainable_parameter_names),trainable_tensors=len(actual),
        trainable_parameters=expected_count,head_gradient_norms=gradients,head_frozen=(lr==0),base_loss=raw['base_loss'],first_loss=first,
        uninterrupted_second_loss=uninterrupted_loss,resumed_second_loss=resumed_loss,max_parameter_difference=maximum,
        scheduler_T_max=sched.T_max,amp_overflow_skips=overflow_skips,successful_update_count=len(successful_updates),
        successful_updates=successful_updates,checkpoint_sha256=sha(folder/'resume.pt'),checkpoint_role='synthetic_resume_only')
    (folder/'resume.pt').unlink() # Only this synthetic, non-candidate temporary file.
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');shutil.copyfile(config_path,folder/'config.yaml')
    (folder/'command.txt').write_text('canonical run_p1.py --smoke\n')
    (folder/'metrics.json').write_text(json.dumps(dict(selected_checkpoint='receipt.json',smoke_only=True))+'\n')
    print(json.dumps(receipt),flush=True);return 0
