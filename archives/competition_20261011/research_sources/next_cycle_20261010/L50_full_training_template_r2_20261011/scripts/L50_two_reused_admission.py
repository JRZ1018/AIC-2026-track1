"""Two disclosed reused-query actual full-combo gates; no old independent gate reuse."""
from pathlib import Path
import json,hashlib,math,yaml
FIRST_SOURCE='4b3f0fe90d8ef5b138d05714526bc92e46dabbdd9254ae849416db61e16638fd'
SECOND_SOURCE='7699f9800d9bfff3265e7598c5ef5998e3bab8790ea6392da0fb67856c3365f0'
SPLITS={'first':'57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7','second':'610111665e1055bcda7157add15b16ade353138365e54ee87dc92316e6dfd6f1'}
PRIMARY='candidate_fixed_graph_P7_vs_actual_L35_GRAPH_P7'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def validate(value):
 need(value.get('schema')=='aictrack1.L50.two_reused_queries.v1' and value.get('status')=='complete_positive_two_reused_queries','two actual L50 reused queries required; L35/B/CG/PF/alpha gates invalid')
 for key,want in [('selected_arm','L50'),('selected_route','L50_GRAPH'),('independent_confirmation',False),('both_queries_previously_disclosed',True),('test_access',False),('used_test_or_online_scores',False),('root_risk_reviewed',True),('visual_learning_rate',5e-6),('head_learning_rate',5e-6),('checkpoint_epoch',20),('checkpoint_role','last'),('final_model','single')]:need(value.get(key)==want,'two-query identity '+key)
 need(set(value.get('queries',{}))=={'first','second'},'exactly two reused queries')
 for name,q in value['queries'].items():
  need(q.get('split_sha256')==SPLITS[name] and q.get('primary_scope')==PRIMARY,'actual L35 combo/query split identity')
  delta=q.get('primary_delta_pp');need(isinstance(delta,(int,float)) and math.isfinite(delta) and delta>0,'both complete combos positive')
  natural=q.get('natural_delta_pp');need(isinstance(natural,(int,float)) and math.isfinite(natural),'natural diagnostic finite only')
  need(q.get('training_seed')==(20261008 if name=='first' else 20261010),'matched existing-control seed')
  need(q.get('training_source_sha256')==(FIRST_SOURCE if name=='first' else SECOND_SOURCE),'actual producer source')
 return value
def actual_ref(ref):
 p=Path(ref['path']);need(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p)==ref['sha256'],'actual evidence path/SHA');return p,read(p)
def verify(value):
 validate(value)
 for name,q in value['queries'].items():
  evidence=q.get('evidence',{});need({'paired_report','result_manifest','root_review','training_source','candidate_audit','candidate_identity','control_audit','numpy_review'}<=set(evidence),'two real paired/review/source/audits/identity/NumPy evidence chains')
  loaded={k:actual_ref(v) for k,v in evidence.items()}
  rp,report=loaded['paired_report'];source_path,source=loaded['training_source'];audit=loaded['candidate_audit'][1];identity=loaded['candidate_identity'][1]
  need(report.get('complete') is True and report.get('primary_scope')==PRIMARY and report.get('independent_confirmation') is False and report.get('test_access') is False and report.get('used_test_or_online_scores') is False,'actual complete disclosed non-test L50 combo')
  need(report.get('schema')==('aictrack1.learning_rate.exploration_paired_graph.v1' if name=='first' else 'aictrack1.L50.second_reused.paired.v1'),'specific actual paired report schema')
  need(report.get('arm',report.get('selected_arm'))=='L50' and report.get('training_source_manifest_sha256')==q['training_source_sha256'] and report.get('split_sha256')==q['split_sha256'],'report arm/source/split')
  for key in ['primary_delta_pp','natural_delta_pp']:need(report.get(key)==q[key],'exact paired delta '+key)
  need(report.get('fixed_graph')==dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=math.log(4.)) and report.get('frozen_P7')==dict(temperature=.5,strength=.75,iterations=60),'frozen graph/P7 mathematical protocol')
  need(report.get('model_updates')==0 and report.get('additional_model_forwards')==0,'no graph training/extra forward')
  manifest=loaded['result_manifest'][1];need(manifest.get('complete') is True,'paired export manifest complete')
  for filename,item in manifest['files'].items():
   rel=Path(filename);need(not rel.is_absolute() and '..' not in rel.parts and sha(rp.parent/rel)==item['sha256'],'all actual paired output SHA')
  need(evidence['training_source']['sha256']==q['training_source_sha256'] and source.get('status')=='bound','real bound training source')
  for rel,h in source['files'].items():
   p=Path(rel);need(not p.is_absolute() and '..' not in p.parts and sha(source_path.parent/p)==h,'producer source closure '+rel)
  need(source['files'].get('scripts/train_peft.py')=='0758b989a1bb33ba851e03f9a852cf35b7ad30847ccbfb62cf528f8dc015a765','official original trainer implementation')
  config=source_path.parent/'configs/experiment/p1_L50.yaml';raw=yaml.safe_load(config.read_text())['experiment'];need(raw['training']['visual_learning_rate']==raw['training']['head_learning_rate']==5e-6 and raw['training']['seed']==q['training_seed'],'actual only LR5/head5/seed recipe')
  for key,want in [('complete',True),('checkpoint_epoch',20),('trainable_tensors',154),('trainable_numel',88233966),('optimizer_parameter_groups',[152,2]),('exact_ac_train_id_label_binding',True),('classifier_parameters_restored_equal',True),('model_frozen_eval',True),('inference_optimizer_steps',0),('test_access',False),('training_source_manifest_sha256',q['training_source_sha256']),('split_sha256',q['split_sha256']),('training_seed',q['training_seed'])]:need(audit.get(key)==want,'actual strict candidate audit '+key)
  need(audit.get('checkpoint_sha256')==identity.get('checkpoint_sha256')==report['candidate']['checkpoint_sha256'] and evidence['candidate_identity']['sha256']==report['candidate']['identity_sha256'],'actual last/cache/paired identity binding')
  need(identity.get('training_source_manifest_sha256')==q['training_source_sha256'] and identity.get('independent_confirmation') is False,'actual disclosed candidate identity')
  terminal=identity.get('exploration_phase_context' if name=='first' else 'confirmation_phase_context',{});need(terminal.get('controller_and_all_arm_groups_closed') is True and terminal.get('canonical_idle') is True,'actual full terminal group closure')
  control=loaded['control_audit'][1];need(control.get('complete') is True and control.get('model_arm')=='L35' and control.get('checkpoint_epoch')==20 and control.get('trainable_tensors')==154 and control.get('inference_optimizer_steps')==0 and control.get('checkpoint_sha256')==report['control']['checkpoint_sha256'],'actual matched L35 control audit/last')
  review=loaded['root_review'][1];need(review.get('complete') is True and review.get('root_authorized') is True and review.get('paired_report_sha256')==evidence['paired_report']['sha256'],'actual root risk review tied to report')
  need(review.get('admit_second_reused_query') is True if name=='first' else review.get('decision')=='allow_single_L50_full_training','actual correct phase root decision')
  numpy=loaded['numpy_review'][1];need(numpy.get('complete') is True and numpy.get('all_manifest_SHA_PASS') is True and numpy.get('report_sha256')==evidence['paired_report']['sha256'],'actual independent CPU integer/manifest review')
 return value
