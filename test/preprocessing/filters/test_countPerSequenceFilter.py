import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.filters.CountPerSequenceFilter import CountPerSequenceFilter
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestCountPerSequenceFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = EnvironmentSettings.root_path / "test/tmp/count_per_seq_filter/"
        PathBuilder.build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                         ["ACF", "ACF"],
                                                                         ["ACF", "ACF", "ACF", "ACF"]], path,
                                                                        seq_metadata=[[{"count": 1}, {"count": 2}, {"count": 3}],
                                                                                      [{"count": 4}, {"count": 1}],
                                                                                      [{"count": 5}, {"count": 6}, {"count": None},
                                                                                       {"count": 1}]])[0])

        dataset1 = CountPerSequenceFilter.process(dataset, {"low_count_limit": 2, "remove_without_count": True, "remove_empty_repertoires": False,
                                                            "result_path": path, "batch_size": 4})
        self.assertEqual(2, dataset1.repertoires[0].get_sequence_aas().shape[0])

        dataset2 = CountPerSequenceFilter.process(dataset, {"low_count_limit": 5, "remove_without_count": True, "remove_empty_repertoires": False,
                                                            "result_path": path, "batch_size": 4})
        self.assertEqual(0, dataset2.repertoires[0].get_sequence_aas().shape[0])

        dataset3 = CountPerSequenceFilter.process(dataset, {"low_count_limit": 0, "remove_without_count": True, "remove_empty_repertoires": False,
                                                            "result_path": path, "batch_size": 4})
        self.assertEqual(3, dataset3.repertoires[2].get_sequence_aas().shape[0])

        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                         ["ACF", "ACF"],
                                                                         ["ACF", "ACF", "ACF", "ACF"]], path,
                                                                        seq_metadata = [[{"count": None}, {"count": None}, {"count": None}],
                                                                                        [{"count": None}, {"count": None}],
                                                                                        [{"count": None}, {"count": None}, {"count": None},
                                                                                         {"count": None}]])[0])

        dataset4 = CountPerSequenceFilter.process(dataset, {"low_count_limit": 0, "remove_without_count": True, "remove_empty_repertoires": False,
                                                            "result_path": path, "batch_size": 4})
        self.assertEqual(0, dataset4.repertoires[0].get_sequence_aas().shape[0])
        self.assertEqual(0, dataset4.repertoires[1].get_sequence_aas().shape[0])
        self.assertEqual(0, dataset4.repertoires[2].get_sequence_aas().shape[0])

        self.assertRaises(AssertionError, CountPerSequenceFilter.process, dataset, {"low_count_limit": 10, "remove_without_count": True,
                                                                                    "remove_empty_repertoires": True, "result_path": path, "batch_size": 4})

        shutil.rmtree(path)
