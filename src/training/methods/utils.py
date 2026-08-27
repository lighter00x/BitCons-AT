from contextlib import contextmanager

from torch.nn.modules.batchnorm import _BatchNorm


@contextmanager
def freeze_batchnorm_stats(model):
    """Run auxiliary forwards without updating BatchNorm running statistics."""
    batchnorm_states = {
        module: module.training
        for module in model.modules()
        if isinstance(module, _BatchNorm)
    }
    try:
        for module in batchnorm_states:
            module.eval()
        yield
    finally:
        for module, was_training in batchnorm_states.items():
            module.train(was_training)


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
