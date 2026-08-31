import torch.nn as nn
from .trades import trades_loss, kl_div
from .mart import mart_loss
from .consistency import consistency_loss
from .bitcons import (
    apply_bitplane_mask,
    apply_unreliable_bitplane_mask,
    bitcons_align_loss,
    bitcons_feature_contrastive_loss,
    get_bitcons_weight,
)


def get_criterion(config):
    if config.method in ('pgd_at', 'bitmax_at', 'bitplane_at', 'quantize_at'):
        return nn.CrossEntropyLoss()
    elif config.method == 'trades':
        return nn.CrossEntropyLoss()
    elif config.method == 'mart':
        return nn.CrossEntropyLoss()
    elif config.method == 'cons_at':
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()
