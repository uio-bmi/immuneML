import os
import shutil
from pathlib import Path
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.distance_encoding.CompAIRRDistanceEncoder import CompAIRRDistanceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestCompAIRRDistanceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, path: str) -> RepertoireDataset:
        repertoires, metadata = RepertoireBuilder.build([["A", "Q"], ["Q", "C"], ["D"], ["E", "F"],
                                                       ["A", "Q"], ["Q", "C"], ["D"], ["E", "F"]], path,
                                                      {"l1": [1, 0, 1, 0, 1, 0, 1, 0], "l2": [2, 3, 2, 3, 2, 3, 3, 3]})
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        return dataset

    def test_encode(self):
        working = 0
        for compairr_path in EnvironmentSettings.compairr_paths:
            if compairr_path.exists():
                working += 1
                self._run_test(compairr_path)
                break

        assert working > 0

    def _run_test(self, compairr_path):

        path = EnvironmentSettings.tmp_test_path / "compairr_distance_encoder/"

        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        enc = CompAIRRDistanceEncoder.build_object(dataset, **{"compairr_path": compairr_path,
                                                           "keep_compairr_input": True,
                                                        "differences": 0,
                                                        "indels": False,
                                                        "ignore_counts": False,
                                                        "threads": 8,
                                                        "ignore_genes": False})

        enc.set_context({"dataset": dataset})
        encoded = enc.encode(dataset, EncoderParams(result_path=path,
                                                    label_config=LabelConfiguration([Label("l1", [0, 1]), Label("l2", [2, 3])]),
                                                    pool_size=4, filename="dataset.pkl"))

        self.assertEqual(8, encoded.encoded_data.examples.shape[0])
        self.assertEqual(8, encoded.encoded_data.examples.shape[1])

        self.assertEqual(0, encoded.encoded_data.examples[0, 0])
        self.assertEqual(0, encoded.encoded_data.examples[1, 1])
        self.assertEqual(0, encoded.encoded_data.examples[0, 4])

        self.assertTrue(np.array_equal([1, 0, 1, 0, 1, 0, 1, 0], encoded.encoded_data.labels["l1"]))
        self.assertTrue(np.array_equal([2, 3, 2, 3, 2, 3, 3, 3], encoded.encoded_data.labels["l2"]))

        shutil.rmtree(path)
