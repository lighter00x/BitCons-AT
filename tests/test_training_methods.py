import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.methods.cons_at import cons_at_train
from training.methods.bitmax_at import bitmax_at_train
from training.methods.bitcons_at import bitcons_at_train
from training.methods.mart import mart_train
from training.methods.pgd_at import pgd_at_train
from training.methods.rpat import rpat_train
from training.methods.trades import trades_train
from training.perturbations import get_perturbation


class TinyClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, inputs, return_feat=False):
        features = self.features(inputs).flatten(1)
        logits = self.classifier(features)
        if return_feat:
            return logits, features
        return logits


class ConfigNamespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def make_config(ce_weight=1.0, align_weight=1.0, contrast=False):
    return ConfigNamespace(
        epsilon=8,
        alpha=2,
        n_steps=1,
        beta=6.0,
        lam=1.0,
        temperature=0.5,
        RA_start=0,
        RA_ip_rate=0.5,
        bitcons=True,
        bitcons_planes=[0, 1, 2],
        bitcons_align='kl',
        bitcons_alpha=1.0,
        bitcons_ce_weight=ce_weight,
        bitcons_align_weight=align_weight,
        bitcons_warmup=0,
        bitcons_warmup_schedule='linear',
        bitcons_start_epoch=0,
        bitcons_gain_tau=0.5,
        bitcons_margin_threshold=-100.0,
        bitcons_contrast=contrast,
        bitcons_contrast_lam=1.0,
        bitcons_contrast_temp=0.5,
        bitmax_planes=[0, 1, 2],
        bitmax_candidates=2,
        bitmax_refine_steps=1,
        bitmax_family_search=False,
        bitmax_refine_best_only=False,
        bitmax_bit_view='selected',
        bitcons_risk_mode='gain',
        bitcons_discrepancy_tau=0.01,
        bitcons_normalize_discrepancy_loss=False,
        bitcons_conflict_mode='none',
        bitcons_conflict_scale=0.1,
        bitcons_max_loss_ratio=1.0,
        perturbation='none',
        awp={'gamma': 0.01, 'warmup': 0},
        rwp={'gamma': 0.01, 'warmup': 0},
        awp_lr=0.01,
        rwp_lr=0.01,
    )


class TrainingMethodsSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = torch.device('cpu')
        cls.images = torch.rand(2, 3, 8, 8)
        cls.labels = torch.tensor([0, 1])
        cls.criterion = nn.CrossEntropyLoss()

    def run_method(self, train_fn, config, paired=False, perturbation_name=None):
        torch.manual_seed(7)
        model = TinyClassifier().to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        perturbation = None
        if perturbation_name is not None:
            config.perturbation = perturbation_name
            perturbation = get_perturbation(
                config, model, optimizer, device=self.device
            )
        if paired:
            images = [self.images.clone(), self.images.flip(-1).clone()]
        else:
            images = self.images.clone()
        loader = [(images, self.labels.clone())]

        loss, accuracy, loss_components = train_fn(
            config,
            model,
            self.device,
            loader,
            optimizer,
            self.criterion,
            perturbation=perturbation,
            epoch=1,
        )

        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)
        self.assertTrue(math.isfinite(loss_components['host_loss']))
        if loss_components.get('bitcons_weighted_loss') is not None:
            self.assertTrue(
                math.isfinite(loss_components['bitcons_weighted_loss'])
            )
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())

    def test_all_bitcons_ablation_combinations_train(self):
        for ce_weight in (0.0, 1.0):
            for align_weight in (0.0, 1.0):
                for contrast in (False, True):
                    with self.subTest(
                        ce_weight=ce_weight,
                        align_weight=align_weight,
                        contrast=contrast,
                    ):
                        self.run_method(
                            pgd_at_train,
                            make_config(ce_weight, align_weight, contrast),
                        )

    def test_full_bitcons_trains_for_supported_methods(self):
        for train_fn in (trades_train, mart_train, rpat_train):
            with self.subTest(method=train_fn.__name__):
                self.run_method(train_fn, make_config(contrast=True))

    def test_cons_at_core_bitcons_trains(self):
        self.run_method(cons_at_train, make_config(contrast=False), paired=True)

    def test_bitmax_at_trains(self):
        config = make_config(contrast=False)
        config.bitcons = False
        self.run_method(bitmax_at_train, config)

    def test_risk_adaptive_bitcons_at_trains(self):
        config = make_config(contrast=False)
        config.bitcons_alpha = 0.05
        self.run_method(bitcons_at_train, config)

    def test_bitcons_at_supports_bitmax_only_ablation(self):
        config = make_config(contrast=False)
        config.bitcons = False
        self.run_method(bitcons_at_train, config)

    def test_family_discrepancy_bitcons_at_trains(self):
        config = make_config(contrast=False)
        config.bitmax_family_search = True
        config.bitmax_refine_best_only = True
        config.bitmax_bit_view = 'best_bit'
        config.bitcons_risk_mode = 'discrepancy'
        config.bitcons_normalize_discrepancy_loss = True
        self.run_method(bitcons_at_train, config)

    def test_conflict_safe_bitcons_at_trains_and_logs_diagnostics(self):
        config = make_config(contrast=False)
        config.bitmax_family_search = True
        config.bitmax_refine_best_only = True
        config.bitmax_bit_view = 'best_bit'
        config.bitcons_risk_mode = 'discrepancy'
        config.bitcons_normalize_discrepancy_loss = True
        config.bitcons_conflict_mode = 'suppress'

        torch.manual_seed(7)
        model = TinyClassifier().to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss, _, components = bitcons_at_train(
            config,
            model,
            self.device,
            [(self.images.clone(), self.labels.clone())],
            optimizer,
            self.criterion,
            epoch=1,
        )

        self.assertTrue(math.isfinite(loss))
        self.assertTrue(math.isfinite(components['bitcons_gradient_cosine']))
        self.assertGreaterEqual(components['bitcons_conflict_rate'], 0.0)
        self.assertLessEqual(components['bitcons_conflict_rate'], 1.0)
        self.assertGreaterEqual(components['bitcons_aux_scale'], 0.0)
        self.assertLessEqual(components['bitcons_aux_scale'], 1.0)

    def test_weight_perturbations_train_with_full_bitcons(self):
        for perturbation_name in ('awp', 'rwp'):
            with self.subTest(perturbation=perturbation_name):
                self.run_method(
                    pgd_at_train,
                    make_config(contrast=True),
                    perturbation_name=perturbation_name,
                )


if __name__ == '__main__':
    unittest.main()
