import os
import shutil
from unittest import TestCase

import numpy as np
from pathlib import Path

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.distance_encoding.EditDistanceEncoder import EditDistanceEncoder
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
        repertoires, metadata = RepertoireBuilder.build([["A", "G"], ["G", "C"], ["D"], ["E", "F"],
                                                       ["A", "G"], ["G", "C"], ["D"], ["E", "F"]], path,
                                                      {"l1": [1, 0, 1, 0, 1, 0, 1, 0], "l2": [2, 3, 2, 3, 2, 3, 3, 3]})
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        return dataset

    def test_encode(self):
        # todo: automatically determine path
        matchairr_path = Path("/Users/lonneke/Programs/vdjsearch/bin/vdjsearch")

        if matchairr_path.exists():
            self._run_test(matchairr_path)

    def _run_test(self, compairr_path):

        path = EnvironmentSettings.tmp_test_path / "distance_encoder/"

        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        enc = EditDistanceEncoder.build_object(dataset, **{"compairr_path": compairr_path,
                                                        "differences": 0,
                                                        "indels": False,
                                                        "ignore_frequency": False,
                                                        "ignore_genes": False})

        enc.set_context({"dataset": dataset})
        encoded = enc.encode(dataset, EncoderParams(result_path=path,
                                                    label_config=LabelConfiguration([Label("l1", [0, 1]), Label("l2", [2, 3])]),
                                                    pool_size=4, filename="dataset.pkl"))

        self.assertEqual(8, encoded.encoded_data.examples.shape[0])
        self.assertEqual(8, encoded.encoded_data.examples.shape[1])

        self.assertEqual(0, encoded.encoded_data.examples.iloc[0, 0])
        self.assertEqual(0, encoded.encoded_data.examples.iloc[1, 1])
        self.assertEqual(0, encoded.encoded_data.examples.iloc[0, 4])

        self.assertTrue(np.array_equal([1, 0, 1, 0, 1, 0, 1, 0], encoded.encoded_data.labels["l1"]))
        self.assertTrue(np.array_equal([2, 3, 2, 3, 2, 3, 3, 3], encoded.encoded_data.labels["l2"]))

        shutil.rmtree(path)
