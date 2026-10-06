from types import SimpleNamespace
import unittest
import torch
from aictrack1.methods.prefix_plasticity import prefix_parameter_groups

class Toy(torch.nn.Module):
    def __init__(self, start=0):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(12)])
        self.stem = torch.nn.Linear(2, 2)
        for n, p in self.named_parameters():
            p.requires_grad_(n.startswith('transformer.resblocks.') and int(n.split('.')[2]) >= start)

class PrefixTests(unittest.TestCase):
    def test_control_exact_group(self):
        model = Toy(4)
        groups = prefix_parameter_groups(model, 1.75e-6)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['lr'], 1.75e-6)
        self.assertEqual([id(p) for p in groups[0]['params']], [id(p) for p in model.parameters() if p.requires_grad])
    def test_candidate_lr_and_coverage(self):
        model = Toy()
        groups = prefix_parameter_groups(model, 1.75e-6, prefix_scale=.1)
        self.assertEqual([len(g['params']) for g in groups], [8, 16])
        self.assertAlmostEqual(groups[0]['lr'], 1.75e-7, places=20)
        self.assertEqual(groups[1]['lr'], 1.75e-6)
        opt = torch.optim.AdamW(groups, weight_decay=.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
        opt.step(); sch.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'] / opt.param_groups[1]['lr'], .1)
    def test_reject_frozen_prefix(self):
        with self.assertRaises(ValueError): prefix_parameter_groups(Toy(4), 1e-6, prefix_scale=.1)
    def test_reject_unfrozen_stem(self):
        model = Toy(); model.stem.weight.requires_grad_(True)
        with self.assertRaises(ValueError): prefix_parameter_groups(model, 1e-6, prefix_scale=.1)
    def test_reject_new_grid_value(self):
        with self.assertRaises(ValueError): prefix_parameter_groups(Toy(), 1e-6, prefix_scale=.2)

if __name__ == '__main__': unittest.main()
