import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from common.args import get_args
from common.config import Config
from losses.bitcons import get_bitcons_weight
from eval import load_exp_config


def load_config(*cli_args):
    original_argv = sys.argv
    try:
        sys.argv = ['train.py', *cli_args]
        args = get_args()
        config = Config()
        config.load_from_args(args)
        return config
    finally:
        sys.argv = original_argv


class ConfigSemanticsTest(unittest.TestCase):
    def test_yaml_defaults_are_preserved(self):
        config = load_config('--config', 'pgd_at')

        self.assertFalse(config.bitcons)
        self.assertFalse(config.bitcons_contrast)
        self.assertEqual(config.bitcons_ce_weight, 1.0)
        self.assertEqual(config.bitcons_align_weight, 1.0)
        self.assertEqual(config.bitcons_planes, [0, 1, 2])
        self.assertEqual(config.bitcons_alpha, 0.25)
        self.assertEqual(config.bitcons_warmup, 60)
        self.assertEqual(config.bitcons_contrast_lam, 0.001)
        self.assertTrue(config.pin_memory)

    def test_bitcons_and_contrast_can_be_enabled_explicitly(self):
        config = load_config(
            '--config', 'pgd_at', '--bitcons', '--bitcons_contrast'
        )

        self.assertTrue(config.bitcons)
        self.assertTrue(config.bitcons_contrast)

    def test_contrast_requires_bitcons(self):
        with self.assertRaisesRegex(ValueError, 'requires --bitcons'):
            load_config('--config', 'pgd_at', '--bitcons_contrast')

    def test_cons_at_rejects_unimplemented_contrast(self):
        with self.assertRaisesRegex(ValueError, 'not implemented for cons_at'):
            load_config('--config', 'cons_at', '--bitcons', '--bitcons_contrast')

    def test_awp_cli_values_update_the_values_used_by_awp(self):
        config = load_config(
            '--config', 'pgd_at',
            '--perturbation', 'awp',
            '--awp_gamma', '0.123',
            '--awp_warmup', '7',
        )

        self.assertEqual(config.awp, {'gamma': 0.123, 'warmup': 7})

    def test_zero_bitcons_weights_remain_zero(self):
        config = load_config(
            '--config', 'pgd_at', '--bitcons', '--bitcons_alpha', '0'
        )

        self.assertEqual(get_bitcons_weight(config, 0), 0.0)
        self.assertEqual(get_bitcons_weight(config, 109), 0.0)

    def test_bitmax_defaults_and_cli_overrides(self):
        config = load_config('--config', 'bitmax_at')
        self.assertEqual(config.method, 'bitmax_at')
        self.assertEqual(config.bitmax_planes, [0, 1, 2])
        self.assertEqual(config.bitmax_candidates, 2)
        self.assertEqual(config.bitmax_refine_steps, 2)
        self.assertFalse(config.bitcons)

        overridden = load_config(
            '--config', 'bitmax_at',
            '--bitmax_planes', '0', '1',
            '--bitmax_candidates', '4',
        )
        self.assertEqual(overridden.bitmax_planes, [0, 1])
        self.assertEqual(overridden.bitmax_candidates, 4)

    def test_bitplane_bpda_defaults_and_override(self):
        config = load_config('--config', 'bitplane_at')
        self.assertEqual(config.method, 'bitplane_at')
        self.assertEqual(config.bitplane_planes, [0, 1, 2])
        self.assertFalse(config.bitcons)

        overridden = load_config(
            '--config', 'bitplane_at', '--bitplane_planes', '0', '1'
        )
        self.assertEqual(overridden.bitplane_planes, [0, 1])

    def test_quantize_at_is_an_explicit_control(self):
        config = load_config('--config', 'quantize_at')

        self.assertEqual(config.method, 'quantize_at')
        self.assertFalse(config.bitcons)
        self.assertFalse(config.bitcons_contrast)
        self.assertIsNone(config.bitplane_planes)

    def test_ablation_weights_accept_zero(self):
        config = load_config(
            '--config', 'pgd_at',
            '--bitcons',
            '--bitcons_ce_weight', '0',
            '--bitcons_align_weight', '0',
        )

        self.assertEqual(config.bitcons_ce_weight, 0.0)
        self.assertEqual(config.bitcons_align_weight, 0.0)

    def test_ablation_weights_reject_negative_values(self):
        with self.assertRaisesRegex(ValueError, 'bitcons_ce_weight'):
            load_config(
                '--config', 'pgd_at', '--bitcons',
                '--bitcons_ce_weight', '-1',
            )

    def test_missing_training_config_fails_early(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Training config'):
            load_config('--config', 'does_not_exist')

    def test_warmup_is_a_ramp_duration(self):
        config = load_config(
            '--config', 'pgd_at',
            '--bitcons',
            '--bitcons_alpha', '1',
            '--bitcons_warmup', '100',
        )

        self.assertEqual(get_bitcons_weight(config, 0), 0.0)
        self.assertEqual(get_bitcons_weight(config, 50), 0.5)
        self.assertEqual(get_bitcons_weight(config, 100), 1.0)

    def test_evaluation_prefers_the_saved_training_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Path(temp_dir)
            saved = {
                'dataset': 'tinynet',
                'model': 'resnet18',
                'method': 'pgd_at',
                'num_classes': 200,
                'bitcons': True,
                'bitcons_planes': [3, 4, 5],
            }
            (experiment / 'config.yaml').write_text(yaml.safe_dump(saved))

            config = load_exp_config(experiment)

        self.assertEqual(config.num_classes, 200)
        self.assertEqual(config.bitcons_planes, [3, 4, 5])


if __name__ == '__main__':
    unittest.main()
