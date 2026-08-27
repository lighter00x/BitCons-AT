import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from losses.bitcons import (
    apply_bitplane_mask,
    apply_unreliable_bitplane_mask,
)
from training.methods.utils import freeze_batchnorm_stats


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


class BatchNormIsolationTest(unittest.TestCase):
    def test_auxiliary_forward_keeps_stats_and_affine_gradients(self):
        model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 2, 1))
        model.train()
        batchnorm = model[0]
        inputs = torch.rand(4, 3, 8, 8)
        mean_before = batchnorm.running_mean.clone()
        variance_before = batchnorm.running_var.clone()

        with freeze_batchnorm_stats(model):
            output = model(inputs)
        output.sum().backward()

        self.assertTrue(model.training)
        self.assertTrue(batchnorm.training)
        self.assertTrue(torch.equal(batchnorm.running_mean, mean_before))
        self.assertTrue(torch.equal(batchnorm.running_var, variance_before))
        self.assertIsNotNone(batchnorm.weight.grad)
        self.assertGreater(batchnorm.weight.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
