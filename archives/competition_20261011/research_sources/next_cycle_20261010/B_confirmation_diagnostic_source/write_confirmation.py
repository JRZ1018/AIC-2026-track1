"""Publish an actual positive new610 B confirmation using the exact frozen FULL verifier."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from fresh_diag_contract import ROOT,QUERY_SOURCE_SHA,QUERY_IDENTITY_SHA,SPLIT_SHA,MATH_SOURCE_SHA,MATH_PROTOCOL_SHA,PRIMARY_SCOPE,sha,verify_owned_sources
sys.path.insert(0,str(ROOT/'reference'))
from B_full_route_contract import validate_confirmation,verify_confirmation

def make_confirmation(evidence):
 # FULL's verifier reads every immutable reference and the actual training closed file list.
 loaded={}
 for name,item in evidence.items():
  if not Path(item['path']).is_absolute() or sha(item['path'])!=item['sha256']:raise ValueError('actual evidence path/SHA '+name)
  loaded[name]=json.loads(Path(item['path']).read_text(encoding='utf-8'))
 report=loaded['paired_report'];source=loaded['training_source'];control=loaded['control_audit'];candidate=loaded['candidate_audit']
 arm=report['selected_arm']
 if report.get('diagnostic_source_manifest_sha256')!=verify_owned_sources():raise ValueError('paired report must be from this exact frozen diagnostic source')
 v=dict(schema='aictrack1.next_cycle10.B_confirmation.v1',status='complete_positive_independent_B_confirmation',selected_arm=arm,selected_route=arm+'_GRAPH',primary_scope=PRIMARY_SCOPE,split_sha256=SPLIT_SHA,query_definition_source_sha256=QUERY_SOURCE_SHA,query_identity_sha256=QUERY_IDENTITY_SHA,training_seed=20261010,checkpoint_epoch=20,checkpoint_role='last',selection_rule='fixed_final20',train_count=118958,natural_count=29737,balanced_count=14760,test_access=False,used_test_or_online_scores=False,root_risk_reviewed=True,final_model='student_only',math_source_sha256=MATH_SOURCE_SHA,math_protocol_sha256=MATH_PROTOCOL_SHA,soft_coefficient={'B25':.25,'B50':.5}[arm],primary_delta_pp=report['primary_delta_pp'],natural_delta_pp=report['natural_delta_pp'],training_source_manifest_sha256=evidence['training_source']['sha256'],diagnostic_source_manifest_sha256=report['diagnostic_source_manifest_sha256'],new_H1_checkpoint_sha256=source['h1_checkpoint_sha256'],new_H1_source_sha256=source['source_h1_manifest_sha256'],control_last_sha256=control['last_sha256'],candidate_last_sha256=candidate['last_sha256'],evidence=evidence,natural_nonnegative_hard_gate=False,ci_lower_positive_hard_gate=False,stable_gain_claim=False)
 # Stronger local aliases prevent accidentally blessing another new-source report/audit.
 if report.get('complete') is not True or report.get('independent_confirmation') is not True or report.get('used_test_or_online_scores') is not False or report.get('test_access') is not False or report.get('new_H1_checkpoint_sha256')!=v['new_H1_checkpoint_sha256'] or report.get('new_H1_source_sha256')!=v['new_H1_source_sha256'] or report.get('control_last_sha256')!=v['control_last_sha256'] or report.get('candidate_last_sha256')!=v['candidate_last_sha256']:raise ValueError('actual report complete and current model/head identities')
 for audit in (control,candidate):
  if audit.get('diagnostic_source_manifest_sha256')!=v['diagnostic_source_manifest_sha256'] or audit.get('raw_student_only') is not True or audit.get('teacher_used_for_inference') is not False or audit.get('inference_optimizer_steps')!=0 or audit.get('josrc_soft_state_complete') is not True:raise ValueError('same diagnostic source and raw student/complete method audit')
 if control.get('auxiliary_soft_coefficient')!=0. or control.get('teacher_active_epochs_checked')!=0 or candidate.get('auxiliary_soft_coefficient')!=v['soft_coefficient']:raise ValueError('actual zero-soft control and selected B recipe')
 return verify_confirmation(v)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 names=['paired_report','root_review','training_source','control_audit','candidate_audit']
 for name in names:p.add_argument('--'+name.replace('_','-'),type=Path,required=True);p.add_argument('--'+name.replace('_','-')+'-sha256',required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify_owned_sources()
 evidence={name:dict(path=str(getattr(a,name).resolve()),sha256=getattr(a,name+'_sha256')) for name in names}
 v=make_confirmation(evidence)
 if a.output.exists():raise FileExistsError('actual typed confirmation will not be overwritten')
 with a.output.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(complete=True,confirmation=str(a.output.resolve()),sha256=sha(a.output),primary_delta_pp=v['primary_delta_pp'])));return 0
if __name__=='__main__':raise SystemExit(main())
