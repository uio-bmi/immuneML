import os
import shutil
from unittest import TestCase
import pandas as pd

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.abundance_encoding.KmerAbundanceEncoder import KmerAbundanceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestKmerAbundanceEncoder(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = EnvironmentSettings.tmp_test_path / "abundance_encoder/"
        PathBuilder.build(path)

        dataset = RepertoireBuilder.build_dataset([["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
                                                        labels={"l1": [True, True, False, False]}, path=path)

        encoder = KmerAbundanceEncoder.build_object(dataset, **{
            "p_value_threshold": 0.4, "sequence_encoding": "continuous_kmer", "k":3, "k_left": 0, "k_right": 0,
            "min_gap": 0, "max_gap": 0
        })

        label_config = LabelConfiguration([Label("l1", [True, False], positive_class=True)])

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config))

        self.assertTrue(np.array_equal(np.array([[1, 4], [1, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

        encoder.p_value_threshold = 0.05

        contingency = pd.read_csv(path / "contingency_table.csv")
        p_values = pd.read_csv(path / "p_values.csv")
        relevant_sequences = pd.read_csv(path / "relevant_sequences.csv")

        self.assertListEqual(list(contingency["positive_present"]), [0, 0, 1, 1, 1, 1, 2, 2, 2])
        self.assertListEqual(list(contingency["negative_present"]), [1, 2, 0, 1, 2, 0, 0, 1, 2])
        self.assertListEqual(list(contingency["positive_absent"]), [2, 2, 1, 1, 1, 1, 0, 0, 0])
        self.assertListEqual(list(contingency["negative_absent"]), [1, 0, 2, 1, 0, 2, 2, 1, 0])

        self.assertListEqual([round(val, 1) for val in p_values["p_values"]], [2.0, 1.0, 2.0, 0.8, 1.0, 2.0, 0.2, 0.5, 1.0])
        self.assertListEqual(list(relevant_sequences["k-mer"]), ["III"])

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config))

        self.assertTrue(np.array_equal(np.array([[0, 4], [0, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

        contingency = pd.read_csv(path / "contingency_table.csv")
        p_values = pd.read_csv(path / "p_values.csv")
        relevant_sequences = pd.read_csv(path / "relevant_sequences.csv")

        self.assertListEqual(list(contingency["positive_present"]), [0, 0, 1, 1, 1, 1, 2, 2, 2])
        self.assertListEqual(list(contingency["negative_present"]), [1, 2, 0, 1, 2, 0, 0, 1, 2])
        self.assertListEqual(list(contingency["positive_absent"]), [2, 2, 1, 1, 1, 1, 0, 0, 0])
        self.assertListEqual(list(contingency["negative_absent"]), [1, 0, 2, 1, 0, 2, 2, 1, 0])

        self.assertListEqual([round(val, 1) for val in p_values["p_values"]], [2.0, 1.0, 2.0, 0.8, 1.0, 2.0, 0.2, 0.5, 1.0])
        self.assertListEqual(list(relevant_sequences["k-mer"]), [])

        shutil.rmtree(path)
