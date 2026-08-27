import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.checkpoint import load_checkpoint, save_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_resume_restores_next_epoch_best_metric_and_scheduler(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2], gamma=0.1
        )
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_checkpoint(
                model,
                optimizer,
                epoch=3,
                best_acc=42.5,
                checkpoint_dir=temp_dir,
                scheduler=scheduler,
            )

            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.SGD(
                restored_model.parameters(), lr=0.1
            )
            restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                restored_optimizer, milestones=[2], gamma=0.1
            )
            next_epoch, best_acc = load_checkpoint(
                restored_model,
                restored_optimizer,
                str(Path(temp_dir) / 'best_model.pt'),
                scheduler=restored_scheduler,
            )

        self.assertEqual(next_epoch, 4)
        self.assertEqual(best_acc, 42.5)
        self.assertEqual(
            restored_scheduler.state_dict(), scheduler.state_dict()
        )


if __name__ == '__main__':
    unittest.main()
