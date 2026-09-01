import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from attacks import (
    PGD,
    generate_bit_candidates,
    generate_bit_candidate_family,
    select_bitmax_candidate,
)


class TinyBatchNormClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, 2)

    def forward(self, inputs):
        features = self.pool(self.bn(inputs)).flatten(1)
        return self.fc(features)


class BitMaxTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.clean = torch.rand(4, 3, 8, 8)
        self.adv = (self.clean + torch.empty_like(self.clean).uniform_(
            -8 / 255, 8 / 255
        )).clamp(0, 1)
        self.labels = torch.tensor([0, 1, 0, 1])

    def test_candidates_are_projected_into_the_threat_ball(self):
        candidates = generate_bit_candidates(
            self.adv, self.clean, [0, 1, 2], 8 / 255, count=4
        )

        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            self.assertLessEqual(
                candidate.sub(self.clean).abs().max().item(),
                8 / 255 + 1e-7,
            )
            self.assertGreaterEqual(candidate.min().item(), 0.0)
            self.assertLessEqual(candidate.max().item(), 1.0)

    def test_invalid_noncontiguous_planes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'contiguous low bits'):
            generate_bit_candidates(
                self.adv, self.clean, [0, 2], 8 / 255, count=2
            )

    def test_family_search_covers_each_contiguous_low_bit_width(self):
        candidates = generate_bit_candidate_family(
            self.adv, self.clean, [0, 1, 2], 8 / 255, count=2
        )

        self.assertEqual(len(candidates), 6)
        for candidate in candidates:
            self.assertLessEqual(
                candidate.sub(self.clean).abs().max().item(),
                8 / 255 + 1e-7,
            )

    def test_selection_is_never_weaker_than_the_original_pgd_candidate(self):
        model = TinyBatchNormClassifier().train()
        with torch.no_grad():
            pgd_loss = nn.functional.cross_entropy(
                model.eval()(self.adv), self.labels, reduction='none'
            )
        model.train()

        selected, stats = select_bitmax_candidate(
            model,
            self.clean,
            self.adv,
            self.labels,
            [0, 1, 2],
            8 / 255,
            count=4,
        )
        with torch.no_grad():
            selected_loss = nn.functional.cross_entropy(
                model.eval()(selected), self.labels, reduction='none'
            )

        self.assertTrue(torch.all(selected_loss >= pgd_loss - 1e-7))
        self.assertGreaterEqual(stats['bitmax_loss_gain'], -1e-7)
        self.assertLessEqual(stats['bitmax_delta_linf'], 8 / 255 + 1e-7)

    def test_detailed_selection_reports_per_sample_loss_gain(self):
        model = TinyBatchNormClassifier().train()

        selected, stats, details = select_bitmax_candidate(
            model,
            self.clean,
            self.adv,
            self.labels,
            [0, 1, 2],
            8 / 255,
            count=3,
            return_details=True,
        )

        self.assertEqual(selected.shape, self.adv.shape)
        self.assertEqual(details['selected_indices'].shape, self.labels.shape)
        self.assertEqual(details['loss_gain'].shape, self.labels.shape)
        self.assertEqual(details['candidate_count'], 4)
        self.assertTrue(torch.all(details['loss_gain'] >= -1e-7))
        self.assertAlmostEqual(
            details['loss_gain'].mean().item(),
            stats['bitmax_loss_gain'],
            places=6,
        )

    def test_best_bit_view_is_returned_even_when_pgd_is_stronger(self):
        model = TinyBatchNormClassifier().train()

        selected, stats, details = select_bitmax_candidate(
            model,
            self.clean,
            self.adv,
            self.labels,
            [0, 1, 2],
            8 / 255,
            count=2,
            family_search=True,
            return_best_bit=True,
            return_details=True,
        )

        self.assertEqual(selected.shape, self.adv.shape)
        self.assertTrue(torch.all(details['selected_indices'] > 0))
        self.assertTrue(torch.all(details['selected_bit']))
        self.assertEqual(details['candidate_count'], 7)
        self.assertGreaterEqual(stats['bitmax_selection_rate'], 0.0)
        self.assertLessEqual(stats['bitmax_selection_rate'], 1.0)

    def test_pgd_eval_mode_preserves_batchnorm_and_restores_training(self):
        model = TinyBatchNormClassifier().train()
        mean_before = model.bn.running_mean.clone()
        variance_before = model.bn.running_var.clone()
        tracked_before = model.bn.num_batches_tracked.clone()
        attack = PGD(eps=8 / 255, alpha=2 / 255, steps=2, model_eval=True)

        adversarial = attack(model, self.clean, self.labels)

        self.assertTrue(model.training)
        self.assertTrue(model.bn.training)
        self.assertTrue(torch.equal(model.bn.running_mean, mean_before))
        self.assertTrue(torch.equal(model.bn.running_var, variance_before))
        self.assertTrue(torch.equal(model.bn.num_batches_tracked, tracked_before))
        self.assertLessEqual(
            adversarial.sub(self.clean).abs().max().item(), 8 / 255 + 1e-7
        )


if __name__ == '__main__':
    unittest.main()
