"""One-time local freeze; do not rerun on an already frozen/deployed source."""
import json
from confirmation_diag_contract import ROOT,sha,save_json,SPLIT_SHA,ROWS_SHA
def main():
 path=ROOT/'source_manifest.json'
 if path.exists():raise FileExistsError('already frozen; prepare a separate revision')
 receipt=json.loads((ROOT/'CPU_TEST_RECEIPT.json').read_text())
 if receipt.get('passed')!=11 or receipt.get('failed')!=0 or receipt.get('complete') is not True:raise ValueError('complete actual CPU test receipt required')
 files={};metadata={}
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or 'fixtures' in p.parts or 'artifacts' in p.relative_to(ROOT).parts or p==path or p.name=='fixture_binding.json':continue
  name=p.relative_to(ROOT).as_posix();files[name]=sha(p);metadata[name]=dict(bytes=p.stat().st_size)
 old=ROOT.parent/'p1_diagnostic_source/source_manifest.json';expected='7bb94f9192a6c02ce0928dd2e94aecb879c661f66097bb66ed625224c7d364f6'
 if sha(old)!=expected:raise ValueError('old7bb source changed')
 save_json(path,dict(schema='aictrack1.confirmation.diagnostic_source.v1',revision=1,prepared_only=True,
  split_sha256=SPLIT_SHA,confirmation_rows_sha256=ROWS_SHA,old_diagnostic_source_manifest_sha256=expected,
  frozen_before_fresh_candidate_results=True,training_source_binding='actual fresh prepared_visual manifest required by explicit CLI SHA',
  files=files,file_metadata=metadata,source_inventory_excludes=['__pycache__','tests/fixtures','tests/fixture_binding.json','artifacts','source_manifest.json (self)'],test_access=False))
 print(json.dumps(dict(complete=True,source_manifest_sha256=sha(path),files=len(files),bytes=sum(v['bytes'] for v in metadata.values()))))
if __name__=='__main__':main()
