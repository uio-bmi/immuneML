import os
import shutil
from unittest import TestCase

import pandas as pd

from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.OperationType import OperationType
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.filters.MetadataRepertoireFilter import MetadataRepertoireFilter
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMetadataRepertoireFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = EnvironmentSettings.root_path + "test/tmp/metadata_filter/"
        PathBuilder.build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                       ["ACF", "ACF"],
                                                                       ["ACF", "ACF", "ACF", "ACF"]], path)[0])

        df = pd.DataFrame(data={"key1": [0, 1, 2], "key2": [0, 1, 2]})
        df.to_csv(path+"metadata.csv")

        dataset.metadata_file = path+"metadata.csv"

        dataset1 = MetadataRepertoireFilter.process(dataset, {
            "criteria": {
                "type": OperationType.GREATER_THAN,
                "value": {
                    "type": DataType.COLUMN,
                    "name": "key2"
                },
                "threshold": 1
            },
            "result_path": path
        })

        self.assertEqual(1, dataset1.get_example_count())

        self.assertRaises(AssertionError, MetadataRepertoireFilter.process, dataset, {
            "criteria": {
                "type": OperationType.GREATER_THAN,
                "value": {
                    "type": DataType.COLUMN,
                    "name": "key2"
                },
                "threshold": 10
            },
            "result_path": path
        })

        shutil.rmtree(path)
