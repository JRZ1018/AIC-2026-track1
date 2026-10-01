import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('p21configrecovery',ROOT/'scripts/run_stage2_p21_full_features_config_recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
R={'hard_seconds':1800,'started_unix':100,'deadline_unix':1900}
L="FileNotFoundError: [Errno 2] configs/data/stage2_fold1_336.yaml"
def test_same_deadline_not_reset():assert m.original_deadline(R,L,1800)==1900
@pytest.mark.parametrize('r,l,now',[(R,L,1900),(R,'',1800),({**R,'deadline_unix':2000},L,1800)])
def test_expired_or_unproven_recovery_rejected(r,l,now):
    with pytest.raises(RuntimeError):m.original_deadline(r,l,now)
