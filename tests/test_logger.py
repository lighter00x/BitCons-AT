import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.logger import Logger


class LoggerTest(unittest.TestCase):
    def test_loss_components_are_written_to_a_separate_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(temp_dir, 'experiment')
            logger.log_loss_components(3, {
                'host_loss': 1.25,
                'bitcons_weight': 0.1,
                'bitcons_ce_loss': None,
            })

            with open(logger.loss_components_file, newline='') as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['epoch'], '3')
        self.assertEqual(rows[0]['host_loss'], '1.250000')
        self.assertEqual(rows[0]['bitcons_ce_loss'], 'N/A')


if __name__ == '__main__':
    unittest.main()
