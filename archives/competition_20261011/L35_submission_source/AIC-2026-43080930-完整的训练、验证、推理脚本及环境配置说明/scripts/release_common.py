"""Portable path adapter. Scientific modules are copied byte-for-byte from L35."""
from pathlib import Path
import os, sys, hashlib, json
CODE=Path(__file__).resolve().parents[1]
PACKAGE=CODE.parent
LIBRARY=PACKAGE/'AIC-2026-43080930-可复现的代码与运行说明文档'
MODELS=PACKAGE/'AIC-2026-43080930-可执行模型文件或容器化复现环境'
RESULTS=LIBRARY/'提交结果'
DATA=CODE/'data'/'metadata'
for path in (LIBRARY/'src', LIBRARY/'vendor', CODE/'scripts'/'frozen'):
    sys.path.insert(0,str(path))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
OFFICIAL_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
FINAL_SHA='ebc130838f898e3f5bc5ad50c9b4d05966a7ce0afbe9bc4f08511b4c2f21d9f6'
CSV_SHA='e02a214cc0ba4476501698b75a546d3d29bc5ee1942f5c91dccf6751a78a7d7f'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load_model(checkpoint=None,model_dir=MODELS,device='cpu',verify_final=False):
    import torch
    from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
    from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
    from aictrack1.models.peft import configure_visual_peft,PEFTImageClassifier
    from aictrack1.models.heads import LinearClassifier
    from aictrack1.engine.peft import load_trainable_state_dict
    checkpoint=Path(checkpoint or Path(model_dir)/'final.pt')
    if verify_final and sha(checkpoint)!=FINAL_SHA:raise ValueError('released final checkpoint SHA mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    expected={'competition_stage':'stage2','num_classes':750,'mode':'full_visual',
              'visual_input_resolution':336,'normalize_features':True,'clip_weight_sha256':OFFICIAL_SHA}
    for k,v in expected.items():
        if state.get(k)!=v:raise ValueError('checkpoint contract differs: '+k)
    trainable=state['trainable_state']
    from aictrack1.data.class_mapping import ClassMapping
    if state.get('class_mapping_sha256')!=ClassMapping.from_csv(DATA/'class_mapping.csv').digest():
        raise ValueError('checkpoint/class mapping identity differs')
    if len(trainable)!=154 or sum(v.numel() for v in trainable.values())!=88233966:
        raise ValueError('complete visual152 + classifier2 required')
    if any(not torch.isfinite(v).all().item() for v in trainable.values()):
        raise ValueError('nonfinite model tensor')
    clip,_,_=load_clip_vit_b32(Path(model_dir)/'openai_clip_ViT-B-32.pt',OFFICIAL_SHA,'cpu')
    visual=clip.visual.float();del clip
    enable_interpolated_visual_resolution(visual,target_resolution=336)
    configure_visual_peft(visual,'full_visual')
    model=PEFTImageClassifier(visual,LinearClassifier(512,750),normalize_features=True)
    load_trainable_state_dict(model,trainable)
    for name,p in model.named_parameters():
        if name not in trainable or not torch.equal(p.detach(),trainable[name]):raise ValueError('restore differs: '+name)
        p.requires_grad_(False)
    return model.to(device).eval(),state
