import shutil
from unittest import TestCase

import numpy as np

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.distance_encoding.DistanceEncoder import DistanceEncoder
from source.encodings.distance_encoding.DistanceMetricType import DistanceMetricType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestDistanceEncoder(TestCase):

    def create_dataset(self, path: str) -> RepertoireDataset:
        repertoires, metadata = RepertoireBuilder.build([["A", "B"], ["B", "C"], ["D"], ["E", "F"],
                                                       ["A", "B"], ["B", "C"], ["D"], ["E", "F"]], path,
                                                      {"l1": [1, 0, 1, 0, 1, 0, 1, 0], "l2": [2, 3, 2, 3, 2, 3, 3, 3]})
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        return dataset

    def test_encode(self):
        path = EnvironmentSettings.tmp_test_path + "distance_encoder/"
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        enc = DistanceEncoder.create_encoder(dataset, {"distance_metric": DistanceMetricType.JACCARD.name,
                                                       "attributes_to_match": ["sequence_aas"],
                                                       "pool_size": 4})

        enc.set_context({"dataset": dataset})
        encoded = enc.encode(dataset, EncoderParams(result_path=path,
                                                    label_configuration=LabelConfiguration([Label("l1", [0, 1]), Label("l2", [2, 3])]),
                                                    batch_size=20, filename="dataset.pkl"))

        self.assertEqual(8, encoded.encoded_data.examples.shape[0])
        self.assertEqual(8, encoded.encoded_data.examples.shape[1])

        self.assertEqual(1, encoded.encoded_data.examples.iloc[0, 0])
        self.assertEqual(1, encoded.encoded_data.examples.iloc[1, 1])
        self.assertEqual(1, encoded.encoded_data.examples.iloc[0, 4])

        self.assertTrue(np.array_equal([1, 0, 1, 0, 1, 0, 1, 0], encoded.encoded_data.labels["l1"]))
        self.assertTrue(np.array_equal([2, 3, 2, 3, 2, 3, 3, 3], encoded.encoded_data.labels["l2"]))

        shutil.rmtree(path)
