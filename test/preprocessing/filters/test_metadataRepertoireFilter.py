import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.analysis.criteria_matches.DataType import DataType
from immuneML.analysis.criteria_matches.OperationType import OperationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.MetadataRepertoireFilter import MetadataRepertoireFilter
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestMetadataRepertoireFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = EnvironmentSettings.root_path / "test/tmp/metadata_filter/"
        PathBuilder.build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                       ["ACF", "ACF"],
                                                                       ["ACF", "ACF", "ACF", "ACF"]], path)[0])

        df = pd.DataFrame(data={"key1": [0, 1, 2], "key2": [0, 1, 2]})
        df.to_csv(path/"metadata.csv")

        dataset.metadata_file = path/"metadata.csv"

        dataset1 = MetadataRepertoireFilter(**{
            "criteria": {
                "type": OperationType.GREATER_THAN.name,
                "value": {
                    "type": DataType.COLUMN.name,
                    "name": "key2"
                },
                "threshold": 1
            },
            "result_path": path
        }).process_dataset(dataset, path)

        self.assertEqual(1, dataset1.get_example_count())

        self.assertRaises(AssertionError, MetadataRepertoireFilter(**{
            "criteria": {
                "type": OperationType.GREATER_THAN.name,
                "value": {
                    "type": DataType.COLUMN.name,
                    "name": "key2"
                },
                "threshold": 10
            }
        }).process_dataset, dataset, path)

        shutil.rmtree(path)
