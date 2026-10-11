"""Bind the actual root budget into a new exclusive source; never trains or accesses assets."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,shutil,sys
TEMPLATE=Path(__file__).resolve().parents[1]
EXEC=Path('/root/autodl-tmp/AICtrack1_next_cycle_AFA_CLS_anchored_20261010')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-root',type=Path,required=True);p.add_argument('--budget',type=Path,required=True);p.add_argument('--expected-budget-sha256',required=True);a=p.parse_args()
 if a.execution_root!=EXEC or a.execution_root.exists() or a.execution_root.is_symlink() or a.execution_root.parent.resolve()!=Path('/root/autodl-tmp').resolve():raise ValueError('new exclusive actual AFA root only')
 if sha(a.budget)!=a.expected_budget_sha256:raise ValueError('actual root budget SHA')
 sys.path.insert(0,str(TEMPLATE/'scripts'));from fresh_route_runtime import budget
 b=json.loads(a.budget.read_text());expected=dict(schema='aictrack1.AFA_CLIP_anchored.budget.v1',root_authorized=True,initial_free_bytes=13013950464,capacity_increase_bytes=32212254720,adjusted_initial_free_bytes=45226205184,data_total_bytes=128849018880,disk_max_bytes=32212254720,min_free_bytes=3221225472,atomic_reserve_bytes=2147483648,final_gpu_reserve_hours=4,test_access=False)
 if any(b.get(k)!=v for k,v in expected.items()) or any(type(b.get(k)) not in (int,float) for k in ('phase_start_unix','deadline_unix')) or not 0<b['deadline_unix']-b['phase_start_unix']<=14400:raise ValueError('actual explicitly authorized AFA phase and unchanged capacity ledger')
 m=json.loads((TEMPLATE/'source_manifest.json').read_text());parent=sha(TEMPLATE/'source_manifest.json')
 if m['status']!='unbound_template' or m['budget_sha256'] is not None:raise ValueError('future template only')
 for key,d in m['files'].items():
  q=PurePosixPath(key);s=TEMPLATE.joinpath(*q.parts)
  if q.is_absolute() or '..' in q.parts or chr(92) in key or s.is_symlink() or sha(s)!=d:raise ValueError('registered closed source member')
 EXEC.mkdir(exist_ok=False)
 for key in m['files']:
  t=EXEC/key;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(TEMPLATE/key,t)
 shutil.copyfile(a.budget,EXEC/'inputs/budget.json');m['files']['inputs/budget.json']=sha(EXEC/'inputs/budget.json')
 m.update(status='bound',template_source_manifest_sha256=parent,budget_sha256=a.expected_budget_sha256)
 (EXEC/'source_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
 budget(EXEC)
 print(json.dumps(dict(complete=True,root=str(EXEC),bound_source_sha256=sha(EXEC/'source_manifest.json'),budget_sha256=a.expected_budget_sha256,model_loaded=False,GPU_started=False,test_access=False)))
if __name__=='__main__':main()
