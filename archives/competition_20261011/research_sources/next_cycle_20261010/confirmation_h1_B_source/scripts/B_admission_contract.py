"""CPU contract for one actual positive B exploratory combination; no future values."""
import math,re
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
SCIENTIFIC_TEMPLATE_SHA='8d9c229f645e20a306b11a69bc57a6f4c7141d0c90783bd7d616cb02e580a276'
MATH_SOURCE_SHA='0096f4c8352e28eae4373485c45e4e6e4f54dfa750ae4b4ddeca8bd8a068df18'
MATH_PROTOCOL_SHA='3c273802ce085f2d6e629743e0b120df13bd1761d29f9e91c4ba83908f317f19'
PRIMARY_SCOPE='candidate_fixed_graph_P7_vs_actual_J5_GRAPH_P7'
SOFT_COEFFICIENTS={'B25':.25,'B50':.5}
def require(value,message):
 if not value:raise ValueError(message)
def digest(value):return isinstance(value,str) and bool(re.fullmatch('[0-9a-f]{64}',value))
def validate_B_admission(a):
 require(a.get('schema')=='aictrack1.next_cycle10.B_admission.v1' and a.get('root_authorized') is True and a.get('complete') is True,'root actual B admission required')
 arm=a.get('selected_arm');delta=a.get('exploration_balanced_delta_pp')
 require(arm in SOFT_COEFFICIENTS and a.get('soft_coefficient')==SOFT_COEFFICIENTS[arm],'one actual B25/B50 recipe only; alpha admission forbidden')
 require(isinstance(delta,(float,int)) and not isinstance(delta,bool) and math.isfinite(delta) and delta>0,'actual positive B combo primary balanced delta required')
 require(a.get('primary_scope')==PRIMARY_SCOPE,'must compare fixed graph/P7 combination directly versus actual J5_GRAPH/P7')
 require(a.get('scientific_template_source_sha256')==SCIENTIFIC_TEMPLATE_SHA and a.get('math_source_sha256')==MATH_SOURCE_SHA and a.get('math_protocol_sha256')==MATH_PROTOCOL_SHA,'exact reviewed B scientific/math/protocol identity required')
 for key in ('exploration_paired_sha256','root_risk_review_sha256','exploration_training_source_sha256'):
  require(digest(a.get(key)),'actual B evidence SHA required '+key)
 require(a.get('query_definition_source_sha256')==QUERY_SOURCE_SHA and a.get('new_H1_checkpoint_sha256') is None,'same preregistered unused query and no future H1 SHA')
 require(a.get('final_model')=='student_only' and a.get('test_access') is False,'B teacher cannot export or access test')
 return a
