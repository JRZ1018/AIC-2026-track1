"""CPU metadata/AST/provider verification only; no future source or model binding."""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full-template',type=Path,required=True);a=p.parse_args()
 for r in [ROOT,a.full_template.resolve()]:
  m=json.loads((r/'template_manifest.json').read_text(encoding='utf-8'))
  if m.get('status')!='unbound_template' or m.get('method_family')!='JoSRC_inspired_B':raise ValueError('B CPU template only')
  for key,d in m['files'].items():
   q=Path(key)
   if q.is_absolute() or '..' in q.parts or not (r/q).resolve().is_relative_to(r) or sha(r/q)!=d:raise ValueError('source/path drift '+key)
   if key.endswith('.py'):ast.parse((r/q).read_text(encoding='utf-8-sig'))
 expected=sha(ROOT/'references/UPSTREAM_FULL_TEMPLATE.json')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_full_provider.py'),'--full-template',str(a.full_template),'--expected-template-sha256',expected],env=env,check=True)
 print(json.dumps(dict(complete=True,CPU_templates_only=True,models_opened=False,images_opened=False,GPU_started=False,test_access=False)));return 0
if __name__=='__main__':raise SystemExit(main())
