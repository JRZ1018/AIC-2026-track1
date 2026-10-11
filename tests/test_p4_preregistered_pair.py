from pathlib import Path
import copy
import yaml

ROOT=Path(__file__).resolve().parents[1]

def test_p4_pair_changes_only_gce_q_and_run_identity():
    configs=[yaml.safe_load((ROOT/f'configs/experiment/stage2_p4_gce_fold1_3090_{tag}.yaml').read_text(encoding='utf-8'))['experiment'] for tag in ['q07_control','q05_candidate']]
    assert [x['base_loss']['q'] for x in configs]==[0.7,0.5]
    assert configs[0]['training']==configs[1]['training']
    assert configs[0]['training']['batch_size']==64
    assert configs[0]['training']['epochs']==8
    normalized=[]
    for cfg in configs:
        cfg=copy.deepcopy(cfg)
        for key in ['run_id','output_dir']: cfg.pop(key)
        cfg['base_loss'].pop('q')
        normalized.append(cfg)
    assert normalized[0]==normalized[1]


def test_p4_followup_configs_are_gated_and_use_correct_fold_heads():
    for fold in (2, 3):
        pair=[]
        for tag in ('q07_control', 'q05_candidate'):
            ex=yaml.safe_load((ROOT/f'configs/experiment/stage2_p4_gce_fold{fold}_3090_{tag}.yaml').read_text(encoding='utf-8'))['experiment']
            assert ex['execution_enabled'] is False
            assert f'fold{fold}-c5-h1-336-' in ex['head_checkpoint']
            assert ex['training']['epochs']==8
            pair.append(copy.deepcopy(ex))
        for ex in pair:
            for key in ('run_id','output_dir'): ex.pop(key)
            ex['base_loss'].pop('q')
        assert pair[0]==pair[1]


def test_p4_full_routes_have_fresh_heads_and_fixed_seeds():
    for route,q,seed in [('candidate',0.5,20260921),('fallback',0.7,20260927)]:
        head=yaml.safe_load((ROOT/f'configs/experiment/stage2_p4_{route}_h1_336_full.yaml').read_text(encoding='utf-8'))['experiment']
        peft=yaml.safe_load((ROOT/f'configs/experiment/stage2_p4_{route}_peft_336_full_14ep.yaml').read_text(encoding='utf-8'))['experiment']
        assert head['execution_enabled'] is False and peft['execution_enabled'] is False
        assert head['training_scope']=='full_train_fixed_epochs'
        assert head['expected_train_count']==peft['expected_train_count']==148695
        assert head['training']['epochs']==100 and peft['training']['epochs']==14
        assert head['training']['seed']==peft['training']['seed']==seed
        assert peft['head_checkpoint']==head['output_dir']+'/final.pt'
        assert peft['base_loss']['q']==q
