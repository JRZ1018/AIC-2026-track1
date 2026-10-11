"""Run the submitted single L35 checkpoint with its frozen four-view decoder."""
import argparse,csv,time
from pathlib import Path
import release_common as common
from postprocess import apply,make_submission

def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--test-root',type=Path,help='official Stage2 test JPG directory')
    choice.add_argument('--synthetic-smoke',action='store_true',help='restore model and four-view synthetic forward only')
    p.add_argument('--model-dir',type=Path,default=common.MODELS)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--graph-seconds',type=int,default=3600)
    args=p.parse_args()
    if args.batch_size<1 or args.workers<0 or args.graph_seconds<1:p.error('invalid batch/workers/budget')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('output must be empty')
    import torch
    from aictrack1.evaluation.four_view import PairedViewDataset,fixed_four_view_logits
    from aictrack1.data.indexed_images import IndexedImage
    from aictrack1.data.contracts import TransformConfig
    from aictrack1.data.transforms import build_split_transforms
    from view_features import FourViewFeatureCollector,FEATURE_PROTOCOL
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; select --device cpu')
    model,state=common.load_model(model_dir=args.model_dir,device=device,verify_final=True)
    collector=FourViewFeatureCollector(model.visual)
    if args.synthetic_smoke:
        with torch.inference_mode():
            x=torch.zeros(1,3,336,336,device=device)
            logits=fixed_four_view_logits(model,x,x)[2]
        features=collector.finish(1);collector.close()
        assert logits.shape==(1,750) and torch.isfinite(logits).all() and collector.calls==4
        assert all(not p.requires_grad for p in model.parameters())
        common.write_json(args.output/'synthetic_smoke.json',{'complete':True,'checkpoint_sha256':common.FINAL_SHA,
            'restored_tensors':154,'restored_parameters':88233966,'synthetic_input':[1,3,336,336],
            'logits':[1,750],'features':[1,512],'forward_calls':4,'extra_forward_calls':0,
            'model_updates':0,'test_image_reads':0,'device':str(device),'torch':torch.__version__})
        print('PASS: complete154 restore; synthetic336 four-view logits/features; zero updates')
        return 0
    rows=list(csv.DictReader((common.DATA/'test_manifest.csv').open(encoding='utf-8-sig',newline='')))
    records=[];root=args.test_root.resolve()
    for i,r in enumerate(rows):
        if r['source_split']!='test' or r['internal_class_index'] or r['official_class_id']:raise ValueError('test manifest must be unlabeled')
        path=(root/r['relative_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise FileNotFoundError(path)
        records.append(IndexedImage(i,r['sample_id'],r['relative_path'],path,'test','test',None,None))
    if len(records)!=37444:raise ValueError('expected official 37444 images')
    native=build_split_transforms(TransformConfig(image_size=336,resize_size=336))['validation']
    scaled=build_split_transforms(TransformConfig(image_size=336,resize_size=384))['validation']
    dataset=PairedViewDataset(records,native,scaled,allow_truncated_images=False)
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
        pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    chunks=[];started=time.monotonic()
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                values=fixed_four_view_logits(model,batch['image'].to(device),batch['scaled_image'].to(device))[2]
            chunks.append(values.detach().cpu().half())
            if i%20==0:print(f'four-view batch {i+1}/{len(loader)}',flush=True)
    logits=torch.cat(chunks);features=collector.finish(len(records));collector.close()
    assert collector.calls==4*len(loader)
    names=tuple(r.relative_path for r in records);ids=[r.sample_id for r in records]
    args.output.mkdir(parents=True,exist_ok=True)
    binding=dict(checkpoint_sha256=common.FINAL_SHA,sample_ids=ids,filenames=names,targets=None,
        test_manifest_sha256=common.sha(common.DATA/'test_manifest.csv'),view_protocol='fixed_four_view_scale384',model_parameter_updates=False)
    torch.save(dict(binding,logits=logits),args.output/'fused_logits.pt')
    torch.save(dict(binding,features=features,feature_protocol=FEATURE_PROTOCOL,extra_forward_calls=0),args.output/'four_view_features.pt')
    pred,_,report=apply(logits.float().numpy(),features.numpy(),deadline=time.monotonic()+args.graph_seconds)
    make_submission(pred,names,args.output,{'fixed_graph':report,'model_parameter_updates':0,'wall_seconds':time.monotonic()-started,
        'torch':torch.__version__,'device':str(device),'extra_forward_calls':0})
    print('Complete: '+str(args.output/'pred_results.zip'))
    return 0
if __name__=='__main__':raise SystemExit(main())
