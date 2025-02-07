import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.analysis.criteria_matches.OperationType import OperationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.MetadataFilter import MetadataFilter
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestMetadataRepertoireFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = EnvironmentSettings.tmp_test_path / "metadata_filter/"
        PathBuilder.remove_old_and_build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                       ["ACF", "ACF"],
                                                                       ["ACF", "ACF", "ACF", "ACF"]], path)[0])

        df = pd.DataFrame(data={"key1": [0, 1, 2], "key2": [1, 1, 2]})
        df.to_csv(path/"metadata.csv")

        dataset.metadata_file = path/"metadata.csv"

        dataset1 = MetadataFilter(**{
            "criteria": {
                "type": OperationType.GREATER_THAN.name,
                "column": "key2",
                "threshold": 1
            },
            "result_path": path
        }).process_dataset(dataset, path / 'ex1')

        self.assertEqual(1, dataset1.get_example_count())

        dataset1 = MetadataFilter(**{
            'criteria': {
                'type': OperationType.IN.name,
                'values': [1],
                'column': 'key2'
            }
        }).process_dataset(dataset, path / 'ex2')

        self.assertEqual(2, dataset1.get_example_count())

        shutil.rmtree(path)
