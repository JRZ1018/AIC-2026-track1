"""No real images, GPU, or checkpoint required for full route configuration tests."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('p21full', ROOT/'scripts/run_stage2_p21_full20_4090.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def read(p):
    return yaml.safe_load((ROOT/p).read_text())['experiment']


def test_confirmed_recipe_full_scope_only():
    m.compare_recipe(read(m.PEFT_CONFIG), read('configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml'))


@pytest.mark.parametrize('key,value', [('expected_train_count',118958), ('training_scope','fixed_split_validation'),
    ('head_checkpoint','../../artifacts/experiments/fold3/best.pt'), ('visual_input_resolution',224)])
def test_recipe_rejects_fold_reuse_or_wrong_scope(key,value):
    full = read(m.PEFT_CONFIG); full[key]=value
    with pytest.raises(ValueError):
        m.compare_recipe(full,read('configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml'))


@pytest.mark.parametrize('key,value',[('epochs',14),('batch_size',128),('workers',4),('seed',20261001)])
def test_recipe_rejects_training_drift(key,value):
    full=copy.deepcopy(read(m.PEFT_CONFIG));full['training'][key]=value
    with pytest.raises(ValueError):
        m.compare_recipe(full,read('configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml'))


def test_new_h1_same_full_recipe():
    head=read(m.H1_CONFIG);old=read('configs/experiment/stage2_p5_sce_h1_336_full_a10.yaml')
    assert head == {**old,'run_id':Path(m.H1).name,'output_dir':'../../'+m.H1}


def test_two_frozen_gates_and_bounded_budget():
    assert len(m.GATES)==2 and all(len(h)==64 for h in m.GATES.values())
    assert m.HARD_SECONDS==18000 and m.MAX_ADDED==4*1024**3
