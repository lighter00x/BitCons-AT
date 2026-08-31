import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from losses import apply_bitplane_mask
from models import get_model
from models.bitplane import BitPlaneBPDA, QuantizeBPDA, bitplane_mask_ste


class IdentityBackbone(nn.Module):
    def forward(self, inputs, return_feat=False):
        features = inputs.flatten(1)
        if return_feat:
            return features, features
        return features


class BitPlaneBPDATest(unittest.TestCase):
    def test_forward_is_exact_mask_and_backward_is_identity(self):
        torch.manual_seed(19)
        inputs = torch.rand(2, 3, 4, 4, requires_grad=True)

        outputs = bitplane_mask_ste(inputs, [0, 1, 2])
        expected = apply_bitplane_mask(inputs, [0, 1, 2])
        outputs.sum().backward()

        self.assertTrue(torch.equal(outputs.detach(), expected))
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_wrapper_masks_every_forward_and_supports_features(self):
        model = BitPlaneBPDA(IdentityBackbone(), [0, 1])
        inputs = torch.rand(2, 3, 4, 4)

        logits, features = model(inputs, return_feat=True)
        expected = apply_bitplane_mask(inputs, [0, 1]).flatten(1)

        self.assertTrue(torch.equal(logits, expected))
        self.assertTrue(torch.equal(features, expected))

    def test_quantize_wrapper_rounds_without_clearing_bits(self):
        model = QuantizeBPDA(IdentityBackbone())
        inputs = torch.tensor([[[[0.5017, 0.2501]]]], requires_grad=True)

        outputs = model(inputs)
        expected = (inputs.detach() * 255.0).round() / 255.0
        outputs.sum().backward()

        self.assertTrue(torch.equal(outputs.detach(), expected.flatten(1)))
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_model_factory_wraps_only_bitplane_method(self):
        defended = get_model(SimpleNamespace(
            model='resnet18',
            num_classes=10,
            method='bitplane_at',
            bitplane_planes=[0, 1, 2],
        ))
        baseline = get_model(SimpleNamespace(
            model='resnet18', num_classes=10, method='pgd_at'
        ))
        quantized = get_model(SimpleNamespace(
            model='resnet18', num_classes=10, method='quantize_at'
        ))

        self.assertIsInstance(defended, BitPlaneBPDA)
        self.assertNotIsInstance(baseline, BitPlaneBPDA)
        self.assertIsInstance(quantized, QuantizeBPDA)
        self.assertEqual(quantized.planes, ())


if __name__ == '__main__':
    unittest.main()
