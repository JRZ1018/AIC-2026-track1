"""Observe existing visual forwards, preserving logits and adding zero forwards."""
from __future__ import annotations

import torch
from torch.nn.functional import normalize

FEATURE_PROTOCOL = 'L2_normalize(mean(L2_normalize(raw_visual_each_of_four_views))) in float32 on CPU'


class FourViewFeatureCollector:
    """Production FV2 makes four B calls; a two-2B fixture layout is also tested.

    Hook returns None and only reads a detached copy. In two-2B layout, each
    scale concatenates original first B and exact flip last B. Production never
    changes its four-call engine or invokes a model from inside this collector.
    """
    def __init__(self, visual, *, feature_dim=512, layout='four_calls_B'):
        if layout not in ('four_calls_B', 'two_calls_2B'):
            raise ValueError('unsupported observed forward layout')
        self.feature_dim = feature_dim
        self.layout = layout
        self.pending = []
        self.chunks = []
        self.calls = 0
        self.handle = visual.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if (not isinstance(output, torch.Tensor) or output.ndim != 2
                or output.shape[1] != self.feature_dim or output.shape[0] < 1):
            raise ValueError('visual hook must observe [batch, feature_dim] tensors')
        copied = output.detach().to(device='cpu', dtype=torch.float32).clone()
        if not bool(torch.isfinite(copied).all()):
            raise ValueError('nonfinite visual embedding')
        self.calls += 1
        if self.layout == 'two_calls_2B':
            if len(copied) % 2:
                raise ValueError('two-2B hook requires original/flip batch halves')
            first, flipped = copied.chunk(2, dim=0)
            self.pending.extend([normalize(first, dim=-1), normalize(flipped, dim=-1)])
        else:
            self.pending.append(normalize(copied, dim=-1))
        if len(self.pending) == 4:
            if len({tuple(v.shape) for v in self.pending}) != 1:
                raise ValueError('view batches are misaligned')
            mean = torch.stack(self.pending).mean(dim=0)
            if bool((torch.linalg.vector_norm(mean, dim=1) < 1e-8).any()):
                raise ValueError('four-view mean embedding has zero norm')
            self.chunks.append(normalize(mean, dim=-1))
            self.pending = []
        return None

    def finish(self, expected_rows):
        if self.pending or not self.chunks:
            raise ValueError('incomplete four-view feature groups')
        values = torch.cat(self.chunks)
        if tuple(values.shape) != (expected_rows, self.feature_dim):
            raise ValueError('feature/logit row counts differ')
        if not bool(torch.isfinite(values).all()) or not torch.allclose(
                torch.linalg.vector_norm(values, dim=1), torch.ones(expected_rows), atol=2e-6, rtol=0):
            raise ValueError('final features must be finite normalized float32')
        return values

    def close(self):
        self.handle.remove()
