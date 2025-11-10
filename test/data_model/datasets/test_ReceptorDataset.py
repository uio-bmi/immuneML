import shutil
from unittest import TestCase

import numpy as np

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestReceptorDataset(TestCase):
    def test_get_metadata(self):

        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "receptor_dataset/")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(2, {2: 1.}, {2: 1.},
                                                                   {"l1": {"True": 1.}, "l2": {"2": 1.}}, path)

        self.assertTrue("l1" in dataset.get_label_names())
        self.assertTrue("l2" in dataset.get_label_names())

        self.assertTrue(np.array_equal(['True', 'True'], dataset.get_metadata(['l1'])['l1']))
        self.assertTrue(np.array_equal(['2', '2'], dataset.get_metadata(['l1', 'l2'])['l2']))
        self.assertTrue(dataset.get_locus() == ["TRA", "TRB"])

        shutil.rmtree(path)
