import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.distance_encoding.DistanceEncoder import DistanceEncoder
from immuneML.encodings.distance_encoding.DistanceMetricType import DistanceMetricType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestDistanceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, path: str) -> RepertoireDataset:
        repertoires, metadata = RepertoireBuilder.build([["A", "B"], ["B", "C"], ["D"], ["E", "F"],
                                                       ["A", "B"], ["B", "C"], ["D"], ["E", "F"]], path,
                                                      {"l1": [1, 0, 1, 0, 1, 0, 1, 0], "l2": [2, 3, 2, 3, 2, 3, 3, 3]})
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        return dataset

    def test_encode(self):
        path = EnvironmentSettings.tmp_test_path / "distance_encoder/"
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        enc = DistanceEncoder.build_object(dataset, **{"distance_metric": DistanceMetricType.JACCARD.name,
                                                       "attributes_to_match": ["sequence_aas"],
                                                       "sequence_batch_size": 20})

        enc.set_context({"dataset": dataset})
        encoded = enc.encode(dataset, EncoderParams(result_path=path,
                                                    label_config=LabelConfiguration([Label("l1", [0, 1]), Label("l2", [2, 3])]),
                                                    pool_size=4))

        self.assertEqual(8, encoded.encoded_data.examples.shape[0])
        self.assertEqual(8, encoded.encoded_data.examples.shape[1])

        self.assertEqual(0, encoded.encoded_data.examples.iloc[0, 0])
        self.assertEqual(0, encoded.encoded_data.examples.iloc[1, 1])
        self.assertEqual(0, encoded.encoded_data.examples.iloc[0, 4])

        self.assertTrue(np.array_equal([1, 0, 1, 0, 1, 0, 1, 0], encoded.encoded_data.labels["l1"]))
        self.assertTrue(np.array_equal([2, 3, 2, 3, 2, 3, 3, 3], encoded.encoded_data.labels["l2"]))

        shutil.rmtree(path)
