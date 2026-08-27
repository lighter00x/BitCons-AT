import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from datasets.utils import load_torchvision_dataset


class LocalDataset:
    calls = []

    def __init__(self, download, **kwargs):
        self.__class__.calls.append(download)


class MissingDataset:
    calls = []

    def __init__(self, download, **kwargs):
        self.__class__.calls.append(download)
        if not download:
            raise RuntimeError('missing')


class DatasetLoadingTest(unittest.TestCase):
    def setUp(self):
        LocalDataset.calls.clear()
        MissingDataset.calls.clear()

    def test_local_dataset_does_not_download(self):
        load_torchvision_dataset(LocalDataset, root='/tmp/local-dataset')
        self.assertEqual(LocalDataset.calls, [False])

    def test_missing_dataset_retries_with_download(self):
        load_torchvision_dataset(MissingDataset, root='/tmp/missing-dataset')
        self.assertEqual(MissingDataset.calls, [False, True])


if __name__ == '__main__':
    unittest.main()
