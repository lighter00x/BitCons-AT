import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from losses.bitcons import (
    apply_bitplane_mask,
    apply_unreliable_bitplane_mask,
    bitcons_js_per_sample,
    bitcons_discrepancy_weights,
    bitcons_risk_weights,
    get_risk_adaptive_bitcons_weight,
)
from training.methods.utils import (
    freeze_batchnorm_stats,
    get_classifier_parameters,
    loss_gradient_cosine,
)


class BitPlaneMaskTest(unittest.TestCase):
    def test_low_three_planes_change_8bit_values_by_at_most_seven(self):
        values = torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16)
        images = values / 255.0

        masked = apply_bitplane_mask(images, [0, 1, 2])
        change_in_levels = (images - masked).abs() * 255.0

        self.assertLessEqual(change_in_levels.max().item(), 7.0001)

    def test_low_three_planes_stay_inside_eight_over_255_for_float_inputs(self):
        torch.manual_seed(11)
        images = torch.rand(4, 3, 8, 8)

        masked = apply_bitplane_mask(images, [0, 1, 2])

        self.assertLessEqual(
            (images - masked).abs().max().item(),
            8.0 / 255.0 + 1e-7,
        )

    def test_reliable_and_unreliable_views_reconstruct_quantized_input(self):
        torch.manual_seed(13)
        images = torch.rand(2, 3, 8, 8)
        quantized = (images * 255.0).round() / 255.0

        reliable = apply_bitplane_mask(images, [0, 1, 2])
        unreliable = apply_unreliable_bitplane_mask(images, [0, 1, 2])

        self.assertTrue(
            torch.allclose(reliable + unreliable, quantized, atol=1e-7, rtol=0)
        )


class RiskAdaptiveBitConsTest(unittest.TestCase):
    def test_per_sample_js_is_zero_for_identical_logits(self):
        logits = torch.tensor([[2.0, -1.0], [0.5, 0.25]], requires_grad=True)

        losses = bitcons_js_per_sample(logits, logits)

        self.assertEqual(losses.shape, torch.Size([2]))
        self.assertTrue(torch.allclose(losses, torch.zeros_like(losses), atol=1e-7))

    def test_risk_weights_require_positive_gain_and_reliable_reference(self):
        bit_losses = torch.tensor([1.0, 2.0, 0.25])
        adversarial_losses = torch.tensor([0.5, 1.0, 0.5])
        adversarial_logits = torch.tensor([
            [3.0, 1.0],
            [3.0, 1.0],
            [2.0, 0.0],
        ])
        labels = torch.tensor([0, 1, 0])

        weights, gains, margins, reliable = bitcons_risk_weights(
            bit_losses,
            adversarial_losses,
            adversarial_logits,
            labels,
            gain_tau=0.5,
            margin_threshold=0.0,
        )

        self.assertTrue(torch.allclose(gains, torch.tensor([0.5, 1.0, 0.0])))
        self.assertTrue(torch.equal(reliable, torch.tensor([True, False, True])))
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(margins, torch.tensor([2.0, -2.0, 2.0])))

    def test_discrepancy_weights_require_reliable_reference(self):
        discrepancies = torch.tensor([0.02, 0.02, 0.005])
        adversarial_logits = torch.tensor([
            [3.0, 1.0],
            [3.0, 1.0],
            [2.0, 0.0],
        ])
        labels = torch.tensor([0, 1, 0])

        weights, margins, reliable = bitcons_discrepancy_weights(
            discrepancies,
            adversarial_logits,
            labels,
            discrepancy_tau=0.01,
        )

        self.assertTrue(torch.equal(reliable, torch.tensor([True, False, True])))
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0, 0.0, 0.5])))
        self.assertTrue(torch.allclose(margins, torch.tensor([2.0, -2.0, 2.0])))

    def test_curriculum_respects_start_epoch_and_warmup(self):
        config = SimpleNamespace(
            bitcons_alpha=0.1,
            bitcons_start_epoch=20,
            bitcons_warmup=40,
            bitcons_warmup_schedule='linear',
        )

        self.assertEqual(get_risk_adaptive_bitcons_weight(config, 19), 0.0)
        self.assertEqual(get_risk_adaptive_bitcons_weight(config, 20), 0.0)
        self.assertAlmostEqual(
            get_risk_adaptive_bitcons_weight(config, 40), 0.05
        )
        self.assertEqual(get_risk_adaptive_bitcons_weight(config, 60), 0.1)

    def test_full_batch_reduction_scales_with_gate_prevalence(self):
        per_sample_js = torch.ones(4)
        dense_gate = torch.ones(4)
        sparse_gate = torch.tensor([1.0, 0.0, 0.0, 0.0])

        dense_loss = (dense_gate * per_sample_js).mean()
        sparse_loss = (sparse_gate * per_sample_js).mean()

        self.assertEqual(dense_loss.item(), 1.0)
        self.assertEqual(sparse_loss.item(), 0.25)


class BatchNormIsolationTest(unittest.TestCase):
    def test_auxiliary_forward_keeps_stats_and_affine_gradients(self):
        model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 2, 1))
        model.train()
        batchnorm = model[0]
        inputs = torch.rand(4, 3, 8, 8)
        mean_before = batchnorm.running_mean.clone()
        variance_before = batchnorm.running_var.clone()

        with freeze_batchnorm_stats(model):
            self.assertTrue(batchnorm.training)
            output = model(inputs)
        output.sum().backward()

        self.assertTrue(model.training)
        self.assertTrue(batchnorm.training)
        self.assertTrue(torch.equal(batchnorm.running_mean, mean_before))
        self.assertTrue(torch.equal(batchnorm.running_var, variance_before))
        self.assertIsNotNone(batchnorm.weight.grad)
        self.assertGreater(batchnorm.weight.grad.abs().sum().item(), 0.0)


class GradientConflictTest(unittest.TestCase):
    def test_classifier_gradient_cosine_detects_opposition(self):
        class ClassifierModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(2, 1)

            def forward(self, inputs):
                return self.classifier(inputs)

        model = ClassifierModel()
        output = model(torch.tensor([[1.0, -1.0]])).sum()
        parameters = get_classifier_parameters(model)

        cosine = loss_gradient_cosine(output, -output, parameters)

        self.assertAlmostEqual(cosine, -1.0, places=6)


if __name__ == '__main__':
    unittest.main()
