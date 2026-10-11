"""Synthetic JSON fixtures only; never genuine admission or candidate evidence."""
import json,hashlib
from pathlib import Path
import PF_admission_contract as a

def write(path,value):
 path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');return hashlib.sha256(path.read_bytes()).hexdigest()
def make_admission(folder):
 folder=Path(folder);paired=folder/'SYNTHETIC_PAIRED.json';risk=folder/'SYNTHETIC_RISK.json'
 report=dict(schema='aictrack1.persistent_filter_joint.exploration_paired_graph.v1',complete=True,arm='PF30',alpha=.1,max_ac_weight=.30,persistent_filter_protocol=a.PROTOCOL,primary_scope=a.PRIMARY_SCOPE,primary_delta_pp=.1,natural_delta_pp=-.2,confidence_interval=[-.1,.3],training_source_manifest_sha256=a.EXPLORATION_SOURCE_SHA,diagnostic_source_manifest_sha256='d'*64,test_access=False)
 report_sha=write(paired,report);review=dict(root_authorized=True,admit_independent_confirmation=True,exploration_paired_sha256=report_sha)
 risk_sha=write(risk,review)
 admission=dict(schema='aictrack1.next_cycle10.PF_admission.v1',root_authorized=True,complete=True,selected_arm='PF30',max_ac_weight=.30,persistent_filter_protocol=a.PROTOCOL,hard_alpha=.1,primary_scope=a.PRIMARY_SCOPE,exploration_balanced_delta_pp=.1,exploration_training_source_sha256=a.EXPLORATION_SOURCE_SHA,persistent_filter_source_sha256=a.PF_SOURCE_SHA,persistent_filter_protocol_sha256=a.PF_PROTOCOL_SHA,exploration_diagnostic_source_sha256='d'*64,exploration_paired_sha256=report_sha,root_risk_review_sha256=risk_sha,query_definition_source_sha256=a.QUERY_SOURCE_SHA,new_H1_checkpoint_sha256=None,test_access=False,exploration_paired=dict(path=str(paired.resolve()),sha256=report_sha),root_risk_review=dict(path=str(risk.resolve()),sha256=risk_sha))
 return admission,report,review

def budget(folder):
 return dict(schema='aictrack1.next_cycle10.confirmation_budget.v1',root_authorized=True,phase_start_unix=10.,deadline_unix=20.,original_initial_free_bytes=13013950464,capacity_added_bytes=32212254720,adjusted_initial_free_bytes=45226205184,actual_disk_total_bytes=128849018880,disk_max_bytes=30*1024**3,min_free_bytes=3*1024**3,atomic_reserve_bytes=2*1024**3,final_gpu_reserve_hours=4,capacity_evidence=dict(path=str((Path(folder)/'SYNTHETIC_NOT_OPENED.json').resolve()),sha256='1'*64))
