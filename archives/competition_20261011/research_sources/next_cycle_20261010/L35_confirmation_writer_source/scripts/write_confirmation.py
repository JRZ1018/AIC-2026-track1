"""Write only actual positive independent L35 confirmation with five SHA-bound evidence refs."""
from pathlib import Path
import argparse,json
from L35_full_route_contract import read,sha,require,verify_confirmation,SPLIT_SHA,QUERY_SOURCE_SHA,QUERY_IDENTITY_SHA,PRIMARY_SCOPE
DIAGNOSTIC_SHA='af65db78d5151a286743ed430b2948fbd1625086ba2f0bdeadba77f528c7fe38'
def build(refs):
 required={'paired_report','root_review','training_source','control_audit','candidate_audit'}
 require(set(refs)==required,'five actual immutable refs required')
 evidence={k:dict(path=str(Path(p).resolve()),sha256=sha(Path(p))) for k,p in refs.items()}
 report=read(refs['paired_report'])
 require(report.get('diagnostic_source_manifest_sha256')==DIAGNOSTIC_SHA,'actual reviewed L35 diagnostic source required')
 v=dict(schema='aictrack1.next_cycle10.L35_confirmation.v1',status='complete_positive_independent_L35_confirmation',selected_arm='L35',selected_route='L35_GRAPH',primary_scope=PRIMARY_SCOPE,visual_learning_rate=3.5e-6,training_seed=20261010,checkpoint_epoch=20,checkpoint_role='last',selection_rule='fixed_final20',train_count=118958,natural_count=29737,balanced_count=14760,test_access=False,used_test_or_online_scores=False,root_risk_reviewed=True,final_model='single',split_sha256=SPLIT_SHA,query_definition_source_sha256=QUERY_SOURCE_SHA,query_identity_sha256=QUERY_IDENTITY_SHA,evidence=evidence)
 for k in ['primary_delta_pp','natural_delta_pp','training_source_manifest_sha256','diagnostic_source_manifest_sha256','new_H1_checkpoint_sha256','new_H1_source_sha256','control_last_sha256','candidate_last_sha256']:v[k]=report.get(k)
 return verify_confirmation(v)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['paired_report','root_review','training_source','control_audit','candidate_audit']:p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=build({k:getattr(a,k) for k in ['paired_report','root_review','training_source','control_audit','candidate_audit']})
 with a.output.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(complete=True,receipt=str(a.output),sha256=sha(a.output))))
if __name__=='__main__':main()
