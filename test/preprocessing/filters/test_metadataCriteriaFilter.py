import shutil
from unittest import TestCase

import pandas as pd

from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.OperationType import OperationType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.filters.MetadataFilter import MetadataFilter
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMetadataFilter(TestCase):
    def test_process(self):
        path = EnvironmentSettings.root_path + "test/tmp/clonotypecountfilter/"
        PathBuilder.build(path)
        dataset = RepertoireDataset(filenames=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                       ["ACF", "ACF"],
                                                                       ["ACF", "ACF", "ACF", "ACF"]], path)[0])

        df = pd.DataFrame(data={"key1": [0, 1, 2], "key2": [0, 1, 2]})
        df.to_csv(path+"metadata.csv")

        dataset.metadata_file = path+"metadata.csv"

        dataset1 = MetadataFilter.process(dataset, {
            "criteria": {
                "type": OperationType.GREATER_THAN,
                "value": {
                    "type": DataType.COLUMN,
                    "name": "key2"
                },
                "threshold": 1
            }
        })

        self.assertEqual(1, dataset1.get_repertoire_count())

        shutil.rmtree(path)
