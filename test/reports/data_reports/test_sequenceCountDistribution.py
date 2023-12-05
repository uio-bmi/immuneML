import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.SequenceCountDistribution import SequenceCountDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceCountDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_sequence_counts_seq_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "sequence_counts")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(50, {4: 0.33, 5: 0.33, 7: 0.33}, {"l1": {"a": 0.5, "b": 0.5}}, path / 'dataset')

        scd = SequenceCountDistribution(dataset, path, 1, split_by_label=True, label="l1")

        result = scd.generate_report()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)

    def test_sequence_lengths_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "receptor_counts")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=50,
                                                                   chain_1_length_probabilities={10:1},
                                                                   chain_2_length_probabilities={10:1},
                                                                   labels={"l1": {"a": 0.5, "b": 0.5}}, path=path / 'dataset')

        scd = SequenceCountDistribution(dataset, path, 1, split_by_label=False)

        result = scd.generate_report()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)

    def test_sequence_lengths_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "receptor_counts")

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=10,
                                                                     sequence_count_probabilities={10:0.5, 20: 0.5},
                                                                     sequence_length_probabilities={10:1},
                                                                     labels={"l1": {"a": 0.5, "b": 0.5}}, path=path / 'dataset')

        scd = SequenceCountDistribution(dataset, path, 1, split_by_label=True)

        result = scd.generate_report()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)
