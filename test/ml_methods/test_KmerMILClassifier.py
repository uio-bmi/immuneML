import os
import shutil
from unittest import TestCase

import numpy as np

from source.caching.CacheType import CacheType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.atchley_kmer_encoding.AtchleyKmerEncoder import AtchleyKmerEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.ml_methods.KmerMILClassifier import KmerMILClassifier
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder


class TestKmerMILClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "kmermil/")

        repertoire_count = 10
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=repertoire_count, sequence_count_probabilities={2: 1},
                                                                     sequence_length_probabilities={4: 1}, labels={"l1": {True: 0.5, False: 0.5}},
                                                                     path=path + "dataset/")
        enc_dataset = AtchleyKmerEncoder(2, 1, 1, 'relative_abundance', False).encode(dataset, EncoderParams(path + "result/",
                                                                                                             LabelConfiguration(
                                                                                                                 [Label("l1", [True, False])])))
        cls = KmerMILClassifier(10, -0.0001, 2, False, 1, 0.01, False, True, 8)
        cls.fit(enc_dataset.encoded_data, enc_dataset.encoded_data.labels, ["l1"])

        predictions = cls.predict(enc_dataset.encoded_data, ["l1"])
        self.assertEqual(repertoire_count, len(predictions["l1"]))
        self.assertEqual(repertoire_count, len([pred for pred in predictions["l1"] if isinstance(pred, bool)]))

        predictions_proba = cls.predict_proba(enc_dataset.encoded_data, ["l1"])
        self.assertEqual(repertoire_count, np.rint(np.sum(predictions_proba["l1"])))
        self.assertEqual(repertoire_count, predictions_proba["l1"].shape[0])

        cls.store(path + "model_storage/")

        cls2 = KmerMILClassifier(10, -0.0001, 2, False, 1, 0.01, False, True, 8)
        cls2.load(path + "model_storage/")

        cls2_vars = vars(cls2)
        del cls2_vars["logistic_regression"]
        cls_vars = vars(cls)
        del cls_vars["logistic_regression"]

        for item, value in cls_vars.items():
            if not isinstance(value, np.ndarray):
                self.assertEqual(value, cls2_vars[item])

        model = cls.get_model(["l1"])
        self.assertEqual(vars(cls), model)

        shutil.rmtree(path)
