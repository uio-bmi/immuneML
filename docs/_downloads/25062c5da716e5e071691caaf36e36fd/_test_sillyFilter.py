import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.SillyFilter import SillyFilter
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator


class TestSillyFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _get_mock_repertoire_dataset(self, path):
        # Create a mock RepertoireDataset with 10 repertoires, each containing 50 sequences of length 15,
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=10,
                                                                     sequence_count_probabilities={50: 1},
                                                                     sequence_length_probabilities={15: 1},
                                                                     labels={},
                                                                     path=path)

        return dataset

    def test_process_dataset(self):
        tmp_path = EnvironmentSettings.tmp_test_path / "silly_filter/"

        dataset = self._get_mock_repertoire_dataset(tmp_path / "original_dataset")

        params = {"fraction_to_keep": 0.8}
        filter = SillyFilter.build_object(**params)

        processed_dataset = filter.process_dataset(dataset, tmp_path / "filtered_dataset")

        # 10 original repertoires, keep 80%
        assert len(processed_dataset.repertoires) == 8

        shutil.rmtree(tmp_path)
