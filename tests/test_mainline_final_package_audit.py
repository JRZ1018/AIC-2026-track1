"""Fast CPU rejection checks; does not read official data or models."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('final_audit',
    Path(__file__).resolve().parents[1] / 'scripts/audit_mainline_final_package.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditChecks(unittest.TestCase):
    def test_reject_false_contract(self):
        with self.assertRaisesRegex(ValueError, 'wrong final'):
            audit.require(False, 'wrong final')

    def test_finite_nested_optimizer(self):
        audit.finite_tensors({'state': {1: {'exp_avg': torch.zeros(2)}},
                              'meta': [100, 'full']})

    def test_reject_nonfinite_nested_optimizer(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            audit.finite_tensors({'state': [{'weight': torch.tensor([float('nan')])}]})


if __name__ == '__main__':
    unittest.main()
