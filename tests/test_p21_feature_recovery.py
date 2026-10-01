"""CPU-only budget checks; no image, test, GPU or network access."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('p21featuresrecovery',ROOT/'scripts/run_stage2_p21_full_features_recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_original_deadline_preserved():
    assert m.recovery_deadline({'hard_seconds':1800,'started_unix':100,'deadline_unix':1900},1500)==1900


@pytest.mark.parametrize('receipt,now',[
    ({'hard_seconds':1800,'started_unix':100,'deadline_unix':1900},1900),
    ({'hard_seconds':1800,'started_unix':100,'deadline_unix':1900},1901),
    ({'hard_seconds':1800,'started_unix':100,'deadline_unix':2000},1500),
    ({'hard_seconds':3600,'started_unix':100,'deadline_unix':3700},1500)])
def test_expired_or_expanded_budget_rejected(receipt,now):
    with pytest.raises(RuntimeError):m.recovery_deadline(receipt,now)
