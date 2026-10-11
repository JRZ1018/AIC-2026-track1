"""Single-backbone token readout; original CLIP attention is never replaced."""
import torch
import torch.nn.functional as F
from torch import nn

from .peft import PEFTImageClassifier
from .residual_token_attention import ResidualTokenReadout


def projected_patch_tokens(visual, images):
    captured = []
    def capture(_module, _inputs, output):
        captured.append(output)
    hook = visual.transformer.register_forward_hook(capture)
    try:
        global_features = visual(images)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].ndim != 3:
        raise ValueError('expected exactly one CLIP token forward')
    tokens = captured[0]
    if tokens.shape[0] < 2 or tokens.shape[1] != images.shape[0]:
        raise ValueError('invalid CLIP token axes')
    patches = visual.ln_post(tokens[1:].permute(1, 0, 2)) @ visual.proj
    if patches.shape[0] != global_features.shape[0] or patches.shape[2] != global_features.shape[1]:
        raise ValueError('projected token dimension mismatch')
    return global_features.float(), patches.float()


class TokenAttentionClassifier(PEFTImageClassifier):
    def __init__(self, visual, classifier, *, normalize_features=False, mode='none', seed=20261005):
        super().__init__(visual, classifier, normalize_features=normalize_features)
        if mode not in ('none', 'spatial', 'mean_channel'):
            raise ValueError('unregistered readout mode')
        self.readout_mode = mode
        self.readout = None
        if mode != 'none':
            # Do not shift the baseline RNG/augmentation stream by constructing a query.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.readout = ResidualTokenReadout(int(visual.proj.shape[1]), mode=mode, residual_cap=.10)

    def forward(self, images):
        if self.readout is None:
            return super().forward(images)
        global_features, tokens = projected_patch_tokens(self.visual, images)
        features = self.readout(global_features, tokens).float()
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return self.classifier(features)


def readout_parameter_groups(model):
    if model.readout is None:
        return []
    return [dict(params=[model.readout.query], lr=1e-4, weight_decay=.01),
            dict(params=[model.readout.gain], lr=1e-4, weight_decay=0.)]


def configure_readout(visual, head, *, normalize_features, setting, device):
    if not isinstance(setting, dict) or set(setting) != {'mode', 'initialization_seed'}:
        raise ValueError('explicit readout contract required')
    if setting['initialization_seed'] != 20261005:
        raise ValueError('unfrozen readout initialization seed')
    return TokenAttentionClassifier(visual, head, normalize_features=normalize_features,
        mode=setting['mode'], seed=setting['initialization_seed']).to(device)
