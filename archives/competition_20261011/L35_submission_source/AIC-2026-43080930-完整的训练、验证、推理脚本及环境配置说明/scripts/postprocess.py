"""Fixed graph + P7. No model, labels, parameter search, or updates."""
import argparse,csv,time,zipfile
from pathlib import Path
import release_common as common
from fixed_final_graph import apply_fixed_graph

def apply(scores,features,*,deadline):
    return apply_fixed_graph(scores,features,deadline=deadline)

def make_submission(predictions,filenames,output,metadata):
    from aictrack1.data.class_mapping import ClassMapping
    from aictrack1.submission import SubmissionSpec,build_submission_package
    mapping=ClassMapping.from_csv(common.DATA/'class_mapping.csv')
    if len(filenames)!=37444 or len(set(filenames))!=37444 or len(mapping)!=750:
        raise ValueError('official 37444/750 submission contract differs')
    return build_submission_package(dict(zip(filenames,map(int,predictions),strict=True)),filenames,mapping,output,
        run_metadata=dict(run_id='L35_GRAPH-portable-reproduction',checkpoint_sha256=common.FINAL_SHA,
        config_hash=common.sha(common.CODE/'evidence/full/config.yaml'),
        data_hash=common.sha(common.DATA/'test_manifest.csv'),
        git_commit='unknown_archive',
        inference_command='portable fixed4V/k10-lambda0.3/P7(0.5,0.75,60)',**metadata),
        spec=SubmissionSpec(include_header=False,filename_mode='basename',official_class_id_pattern=r'\d{4}'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=common.CODE/'reproduction_cache')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hard-seconds',type=int,default=3600)
    args=parser.parse_args()
    if args.hard_seconds<1:parser.error('hard-seconds must be positive')
    import torch
    scores=torch.load(args.cache_dir/'fused_logits.pt',map_location='cpu',weights_only=True,mmap=True)
    features=torch.load(args.cache_dir/'four_view_features.pt',map_location='cpu',weights_only=True,mmap=True)
    for field in ('checkpoint_sha256','sample_ids','filenames','test_manifest_sha256','view_protocol'):
        if scores[field]!=features[field]:raise ValueError('cache binding differs: '+field)
    if scores['checkpoint_sha256']!=common.FINAL_SHA or scores['test_manifest_sha256']!=common.sha(common.DATA/'test_manifest.csv'):
        raise ValueError('cache is not the submitted L35 checkpoint/test manifest')
    if scores['targets'] is not None or features['targets'] is not None:
        raise ValueError('unlabeled frozen inference cache required')
    names=tuple(r['relative_path'] for r in csv.DictReader((common.DATA/'test_manifest.csv').open(encoding='utf-8-sig',newline='')))
    if tuple(scores['filenames'])!=names:raise ValueError('test manifest order differs')
    pred,_,report=apply(scores['logits'].float().numpy(),features['features'].float().numpy(),deadline=time.monotonic()+args.hard_seconds)
    make_submission(pred,names,args.output,{'fixed_graph':report,'model_parameter_updates':0,'cache_replay':True})
    actual=common.sha(args.output/'pred_results.csv')
    if actual!=common.CSV_SHA:raise RuntimeError('CSV differs from released submission')
    common.write_json(args.output/'reproduction_check.json',{'complete':True,'csv_byte_identical':True,'csv_sha256':actual,
        'rows':len(names),'classes':750,'test_image_reads':0,'model_forwards':0,'model_updates':0,'fixed_graph':report})
    print('PASS: 37444 rows; CSV bytes exactly match submitted L35')
    return 0
if __name__=='__main__':raise SystemExit(main())
