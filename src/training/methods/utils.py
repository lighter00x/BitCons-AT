from contextlib import contextmanager

import torch
from torch.nn.modules.batchnorm import _BatchNorm


@contextmanager
def freeze_batchnorm_stats(model):
    """Preserve BN running state without changing train-mode computation."""
    batchnorm_states = {
        module: module.track_running_stats
        for module in model.modules()
        if isinstance(module, _BatchNorm)
    }
    try:
        for module in batchnorm_states:
            module.track_running_stats = False
        yield
    finally:
        for module, track_running_stats in batchnorm_states.items():
            module.track_running_stats = track_running_stats


def get_classifier_parameters(model):
    """Return parameters of the model's final classifier."""
    for attribute in ('fc', 'linear', 'classifier'):
        module = getattr(model, attribute, None)
        if module is not None:
            return [parameter for parameter in module.parameters()]
    raise ValueError('Model has no recognized final classifier')


def loss_gradient_cosine(loss_a, loss_b, parameters, eps=1e-12):
    """Measure cosine between two loss gradients on selected parameters."""
    parameters = list(parameters)
    grads_a = torch.autograd.grad(
        loss_a,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    grads_b = torch.autograd.grad(
        loss_b,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    pairs = [
        (grad_a, grad_b)
        for grad_a, grad_b in zip(grads_a, grads_b)
        if grad_a is not None and grad_b is not None
    ]
    if not pairs:
        return 0.0
    dot = sum((grad_a * grad_b).sum() for grad_a, grad_b in pairs)
    norm_a = sum(grad_a.square().sum() for grad_a, _ in pairs).sqrt()
    norm_b = sum(grad_b.square().sum() for _, grad_b in pairs).sqrt()
    return float((dot / (norm_a * norm_b + eps)).detach().item())


class LossComponentMeter:
    """Accumulate raw and weighted training-loss components for one epoch."""

    def __init__(self):
        self.host_total = 0.0
        self.batch_count = 0
        self.aux_batch_count = 0
        self.bitcons_weight_total = 0.0
        self.bitcons_ce_total = 0.0
        self.bitcons_align_total = 0.0
        self.bitcons_contrast_total = 0.0
        self.bitcons_weighted_total = 0.0

    @staticmethod
    def _value(loss):
        return float(loss.detach().item()) if loss is not None else 0.0

    def update(
        self,
        host_loss,
        bitcons_weight=0.0,
        bitcons_ce=None,
        bitcons_align=None,
        bitcons_contrast=None,
        bitcons_weighted=None,
    ):
        self.host_total += self._value(host_loss)
        self.batch_count += 1
        if bitcons_ce is None and bitcons_align is None and bitcons_contrast is None:
            return
        self.aux_batch_count += 1
        self.bitcons_weight_total += float(bitcons_weight)
        self.bitcons_ce_total += self._value(bitcons_ce)
        self.bitcons_align_total += self._value(bitcons_align)
        self.bitcons_contrast_total += self._value(bitcons_contrast)
        self.bitcons_weighted_total += self._value(bitcons_weighted)

    def averages(self):
        result = {
            'host_loss': self.host_total / max(self.batch_count, 1),
            'bitcons_weight': None,
            'bitcons_ce_loss': None,
            'bitcons_align_loss': None,
            'bitcons_contrast_loss': None,
            'bitcons_weighted_loss': None,
        }
        if self.aux_batch_count == 0:
            return result
        count = self.aux_batch_count
        result.update({
            'bitcons_weight': self.bitcons_weight_total / count,
            'bitcons_ce_loss': self.bitcons_ce_total / count,
            'bitcons_align_loss': self.bitcons_align_total / count,
            'bitcons_contrast_loss': self.bitcons_contrast_total / count,
            'bitcons_weighted_loss': self.bitcons_weighted_total / count,
        })
        return result
