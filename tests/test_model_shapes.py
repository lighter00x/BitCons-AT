import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models import get_model


class ModelShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_models_support_cifar_and_tinynet_input_sizes(self):
        for model_name in (
            'resnet18',
            'preactresnet18',
            'wrn28_10',
            'wrn34_10',
        ):
            model = get_model(SimpleNamespace(model=model_name, num_classes=7))
            model.eval()
            for image_size in (32, 64):
                with self.subTest(model=model_name, image_size=image_size):
                    inputs = torch.rand(1, 3, image_size, image_size)
                    with torch.no_grad():
                        logits, features = model(inputs, return_feat=True)
                    self.assertEqual(logits.shape, (1, 7))
                    self.assertEqual(features.ndim, 2)
                    self.assertEqual(features.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
