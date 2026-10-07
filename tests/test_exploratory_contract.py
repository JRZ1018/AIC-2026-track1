import copy
import unittest
from aictrack1.submission.exploratory_contract import (
    MX1_LAST, MX1_SOURCE, MX1_FORMAL, H1, validate_mx1_exploration,
    validate_terminal_history, inference_only_payload,
)


def contract():
    return dict(schema='aictrack1.mx1.exploration.v1', explicit_user_authorization=True,
                authorization_text='MX1 MixUp监督未证实稳定收益，是否可以先出一个探索包，我来测分，看到底是提高还是降低？',
                last_sha256=MX1_LAST, training_source_sha256=MX1_SOURCE,
                formal_report_sha256=MX1_FORMAL, head_sha256=H1,
                checkpoint_selection='fixed last epoch20; never best', train_count=118958,
                validation_count=29737, full_train=False, screen_gate_passed=False,
                multi_training_seed_confirmed=False, maximum_test_runs=1,
                online_submission_by_agent=False, p7=dict(temperature=.5, strength=.75, iterations=60),
                four_view='336+flip+384crop336+flip')


class ExplorationTests(unittest.TestCase):
    def test_explicit_failed_gate_exception(self):
        validate_mx1_exploration(contract())

    def test_reject_scope_promotion_and_parameter_changes(self):
        for key, value in [('full_train', True), ('screen_gate_passed', True),
                           ('multi_training_seed_confirmed', True), ('maximum_test_runs', 2),
                           ('explicit_user_authorization', False), ('online_submission_by_agent', True),
                           ('last_sha256', 'best'), ('training_source_sha256', 'old'),
                           ('head_sha256', 'old'), ('formal_report_sha256', 'old'),
                           ('train_count', 148695), ('p7', dict(temperature=.75, strength=.75, iterations=60))]:
            v = copy.deepcopy(contract()); v[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mx1_exploration(v)

    def test_compact_retains_learned_state_and_protocol(self):
        tensors = {str(i): object() for i in range(96)}
        state = dict(epoch=20, head_initializer_sha256=H1,
                     training_scope='fixed_split_validation', trainable_state=tensors,
                     history=[dict(epoch=i) for i in range(1,21)], optimizer=object(),
                     input_mix_protocol={'alpha': .2}, config_sha256='original')
        result = inference_only_payload(state)
        self.assertIs(result['trainable_state'], tensors)
        self.assertNotIn('optimizer', result)
        self.assertEqual(result['input_mix_protocol'], state['input_mix_protocol'])
        self.assertEqual(result['config_sha256'], 'original')
        self.assertEqual(state['epoch'], 20)

    def test_incomplete_and_wrong_head_rejected(self):
        with self.assertRaises(ValueError):
            validate_terminal_history(dict(epoch=19, head_initializer_sha256=H1))
        with self.assertRaises(ValueError):
            validate_terminal_history(dict(epoch=20, head_initializer_sha256='old'))

    def test_engine_restore_must_precede_inference_freeze(self):
        import torch
        from aictrack1.engine.peft import load_trainable_state_dict
        model = torch.nn.Linear(3, 2)
        saved = {name: value.detach().clone() for name, value in model.named_parameters()}
        load_trainable_state_dict(model, saved)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with self.assertRaises(ValueError):
            load_trainable_state_dict(model, saved)
        self.assertTrue(all(torch.equal(value, saved[name]) for name,value in model.named_parameters()))


if __name__ == '__main__':
    unittest.main()
