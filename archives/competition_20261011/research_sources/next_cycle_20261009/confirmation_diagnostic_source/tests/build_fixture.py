"""Generate production-shaped synthetic assets using the actual confirmation builder."""
import importlib.util,json,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BOOTSTRAP=ROOT.parent/'confirmation_source'
EXISTING=BOOTSTRAP/'test_fixtures/visual_a3dba32b07bc44928b0ec409672d99ff'
sys.path[:0]=[str(BOOTSTRAP/'scripts'),str(BOOTSTRAP/'src')]
spec=importlib.util.spec_from_file_location('actual_confirmation_visual_builder',BOOTSTRAP/'scripts/prepare_confirmation_visual.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
payload=json.loads((EXISTING/'inputs/p1_assets.json').read_text())
selection=EXISTING/'inputs/selection_receipt.json'
selection_sha=builder.sha(selection)
payload.update(selection_receipt_sha256=selection_sha,source_h1_manifest_sha256='2'*64)
target=ROOT/'tests/fixtures'/('visual_'+uuid.uuid4().hex)
builder.create_visual_source(target,payload,'J5',selection,selection_sha,'2'*64)
(ROOT/'tests/fixture_binding.json').write_text(json.dumps(dict(training_root=str(target),training_source_sha256=builder.sha(target/'source_manifest.json'),
 fixture_only=True,trained_h1_verified=False,actual_builder_sha256=builder.sha(BOOTSTRAP/'scripts/prepare_confirmation_visual.py')),indent=2)+'\n')
