"""Read-only CPU provider imports/signatures/parser; no checkpoint, image or CUDA probe."""
from __future__ import annotations
import argparse,ast,hashlib,importlib,inspect,json,os,subprocess,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full-template',type=Path,required=True);parser.add_argument('--expected-template-sha256',required=True);args=parser.parse_args()
    root=args.full_template.resolve();manifest=root/'template_manifest.json'
    if sha(manifest)!=args.expected_template_sha256:raise ValueError('actual reviewed FULL template digest differs')
    value=json.loads(manifest.read_text(encoding='utf-8'))
    if value.get('schema')!='aictrack1.fresh_route.full.training.template.v1':raise ValueError('actual conditional FULL template required')
    for rel,digest in value['files'].items():
        p=root/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or not p.resolve().is_relative_to(root) or sha(p)!=digest:raise ValueError('actual FULL provider drift '+rel)
    sys.path[:0]=[str(root/'scripts'),str(root/'src')]
    contract=importlib.import_module('p1_contract');head=importlib.import_module('full_h1_reference')
    for module in (contract,head):
        if not Path(module.__file__).resolve().is_relative_to(root):raise ValueError('provider module outside reviewed package')
    for name,arguments in [('assets',(root,)),('verify_files',(root,)),('verify_remote_assets',(root,)),('validate_config',({},'B25',{})),('selection',(Path('/not_opened/confirmation.json'),'not_opened')),('verify_selection_evidence',({},))]:inspect.signature(getattr(contract,name)).bind(*arguments)
    inspect.signature(head.load_new_fold_head).bind(Path('/not_opened/head.pt'),feature_dim=512,num_classes=750)
    inspect.signature(head.bind_rows).bind([],[],[],[])
    tree=ast.parse((root/'scripts/train_n1_peft.py').read_text(encoding='utf-8'))
    dictionaries={}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            if any(isinstance(x,ast.Name) and x.id in ('checkpoint','selected_checkpoint','metrics') for x in targets) and isinstance(node.value,ast.Dict):
                dictionaries[next(x.id for x in targets if isinstance(x,ast.Name))]={x.value for x in node.value.keys if isinstance(x,ast.Constant) and isinstance(x.value,str)}
    for kind in ('checkpoint','selected_checkpoint','metrics'):
        if not {'fresh_route_full_source_manifest_sha256','fresh_route_full_arm','selected_route','classifier_head_source','classifier_head_restored_equal','fresh_confirmation_receipt_sha256','full_budget_sha256'}<=dictionaries.get(kind,set()):raise ValueError('actual producer typed fields absent '+kind)
    for script in ('train_n1_peft.py','run_full.py','prepare_full_route.py'):
        done=subprocess.run([sys.executable,'-B',str(root/'scripts'/script),'--help'],capture_output=True,text=True,timeout=30)
        if done.returncode or 'usage:' not in done.stdout:raise RuntimeError('actual FULL CLI import failed '+script+done.stderr)
    print(json.dumps(dict(schema='aictrack1.fresh_route.final.provider_cpu_check.v1',complete=True,full_template_manifest_sha256=args.expected_template_sha256,actual_provider_api_signatures=True,actual_producer_typed_payload_fields=True,actual_cli_help=True,checkpoint_opened=False,model_constructed=False,images_opened=False,server_access=False,cuda_used=False)))
    return 0
if __name__=='__main__':raise SystemExit(main())
