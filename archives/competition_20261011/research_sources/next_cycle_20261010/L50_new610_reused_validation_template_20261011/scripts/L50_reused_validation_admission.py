"""Root-reviewed positive first reused query; never masquerades as L35 admission."""
from pathlib import Path
import hashlib,json
def validate_admission(value):
 if value.get('schema')!='aictrack1.L50.first_reused_query.admission.v1' or value.get('selected_arm')!='L50' or value.get('root_risk_review_complete') is not True or value.get('independent_confirmation') is not False:raise ValueError('explicit actual L50 reused-query admission required')
 for name in ('paired_report','root_review'):
  ref=value[name];p=Path(ref['path'])
  if not p.is_absolute() or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('real admission reference '+name)
 report=json.loads(Path(value['paired_report']['path']).read_text())
 if report.get('complete') is not True or report.get('arm')!='L50' or report.get('training_source_manifest_sha256')!='4b3f0fe90d8ef5b138d05714526bc92e46dabbdd9254ae849416db61e16638fd' or report.get('primary_scope')!='candidate_fixed_graph_P7_vs_actual_L35_GRAPH_P7' or not report.get('primary_delta_pp',0)>0:raise ValueError('actual first reused query complete combo must be positive versus L35')
 review=json.loads(Path(value['root_review']['path']).read_text())
 if review.get('complete') is not True or review.get('admit_second_reused_query') is not True:raise ValueError('actual root risk review required')
 return value
