import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.filters.ClonotypeCountFilter import ClonotypeCountFilter
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestClonotypeCountFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = EnvironmentSettings.root_path + "test/tmp/clonotypecountfilter/"
        PathBuilder.build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                       ["ACF", "ACF"],
                                                                       ["ACF", "ACF", "ACF", "ACF"]], path)[0])

        dataset1 = ClonotypeCountFilter.process(dataset, {"lower_limit": 3, "result_path": path})
        self.assertEqual(2, dataset1.get_example_count())

        dataset2 = ClonotypeCountFilter.process(dataset, {"upper_limit": 2, "result_path": path})
        self.assertEqual(1, dataset2.get_example_count())

        shutil.rmtree(path)
