"""No image/GPU/network access; new allocation needs explicit old failure."""
import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('p21featuresfresh',ROOT/'scripts/run_stage2_p21_full_features_fresh.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RECEIPT={'hard_seconds':1800,'started_unix':100,'deadline_unix':1900}
LOG="python: can't open file '/root/path/scripts/extract_clip_features.py': [Errno 2] No such file"

def test_requires_pre_image_failure_and_new_identity():
    m.verify_pre_image_failure(RECEIPT,LOG)
    assert m.HARD_SECONDS==1800 and m.QUEUE!=m.ORIGINAL and m.QUEUE!=m.RECOVERY

@pytest.mark.parametrize('log',['','CUDA out of memory','epoch_complete',"can't open file scripts/other.py [Errno 2]"])
def test_other_failures_not_implicitly_restarted(log):
    with pytest.raises(RuntimeError):m.verify_pre_image_failure(RECEIPT,log)

def test_old_deadline_not_rewritten():
    with pytest.raises(RuntimeError):m.verify_pre_image_failure({**RECEIPT,'deadline_unix':2000},LOG)
