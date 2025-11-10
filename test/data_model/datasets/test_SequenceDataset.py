import shutil
from unittest import TestCase

import numpy as np

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceDataset(TestCase):
    def test_get_metadata(self):

        path = EnvironmentSettings.tmp_test_path / "sequence_dataset/"
        PathBuilder.build(path)

        dataset = RandomDatasetGenerator.generate_sequence_dataset(2, {2: 1.}, {"l1": {"True": 1.}, "l2": {"2": 1.}}, path)

        self.assertTrue("l1" in dataset.get_label_names())
        self.assertTrue("l2" in dataset.get_label_names())

        self.assertTrue(np.array_equal(['True', 'True'], dataset.get_metadata(['l1'])['l1']))
        self.assertTrue(np.array_equal(['2', '2'], dataset.get_metadata(['l1', 'l2'])['l2']))
        self.assertTrue(dataset.get_locus() == ["TRB"])

        shutil.rmtree(path)
