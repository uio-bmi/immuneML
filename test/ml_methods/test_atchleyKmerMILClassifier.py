import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.atchley_kmer_encoding.AtchleyKmerEncoder import AtchleyKmerEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.ml_methods.AtchleyKmerMILClassifier import AtchleyKmerMILClassifier
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestAtchleyKmerMILClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "kmermil")

        repertoire_count = 10
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=repertoire_count, sequence_count_probabilities={2: 1},
                                                                     sequence_length_probabilities={4: 1}, labels={"l1": {True: 0.5, False: 0.5}},
                                                                     path=path / "dataset")
        enc_dataset = AtchleyKmerEncoder(2, 1, 1, 'relative_abundance', False).encode(dataset, EncoderParams(path / "result",
                                                                                                             LabelConfiguration(
                                                                                                                 [Label("l1", [True, False])])))
        cls = AtchleyKmerMILClassifier(iteration_count=10, threshold=-0.0001, evaluate_at=2, use_early_stopping=False, random_seed=1, learning_rate=0.01,
                                       zero_abundance_weight_init=True, number_of_threads=8)
        cls.fit(enc_dataset.encoded_data, Label("l1"))

        predictions = cls.predict(enc_dataset.encoded_data, Label("l1"))
        self.assertEqual(repertoire_count, len(predictions["l1"]))
        self.assertEqual(repertoire_count, len([pred for pred in predictions["l1"] if isinstance(pred, bool)]))

        predictions_proba = cls.predict_proba(enc_dataset.encoded_data, Label("l1"))
        self.assertEqual(repertoire_count, np.rint(np.sum(predictions_proba["l1"])))
        self.assertEqual(repertoire_count, predictions_proba["l1"].shape[0])

        cls.store(path / "model_storage", feature_names=enc_dataset.encoded_data.feature_names)

        cls2 = AtchleyKmerMILClassifier(iteration_count=10, threshold=-0.0001, evaluate_at=2, use_early_stopping=False, random_seed=1, learning_rate=0.01,
                                        zero_abundance_weight_init=True, number_of_threads=8)
        cls2.load(path / "model_storage")

        cls2_vars = vars(cls2)
        del cls2_vars["logistic_regression"]
        cls_vars = vars(cls)
        del cls_vars["logistic_regression"]

        for item, value in cls_vars.items():
            if not isinstance(value, np.ndarray) and not isinstance(value, Label):
                loaded_value = cls2_vars[item]
                self.assertEqual(value, loaded_value)

        shutil.rmtree(path)
