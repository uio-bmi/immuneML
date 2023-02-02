import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotReceptorEncoder import OneHotReceptorEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.ml_methods.ReceptorCNN import ReceptorCNN
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestReceptorCNN(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "cnn")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=500, chain_1_length_probabilities={4: 1},
                                                                   chain_2_length_probabilities={4: 1},
                                                                   labels={"CMV": {True: 0.5, False: 0.5}}, path=path / "dataset")
        enc_dataset = OneHotReceptorEncoder(True, 1, False, "enc1").encode(dataset, EncoderParams(path / "result",
                                                                                           LabelConfiguration([Label("CMV", [True, False])])))
        cnn = ReceptorCNN(kernel_count=2, kernel_size=[3], positional_channels=3, sequence_type="amino_acid", device="cpu",
                          number_of_threads=4, random_seed=1, learning_rate=0.01, iteration_count=10, l1_weight_decay=0.1, evaluate_at=5,
                          batch_size=100, training_percentage=0.8, l2_weight_decay=0.0)
        cnn.fit(encoded_data=enc_dataset.encoded_data, label=Label("CMV", [True, False]))

        predictions = cnn.predict(enc_dataset.encoded_data, Label("CMV", [True, False]))
        self.assertEqual(500, len(predictions["CMV"]))
        self.assertEqual(500, len([pred for pred in predictions["CMV"] if isinstance(pred, bool)]))

        predictions_proba = cnn.predict_proba(enc_dataset.encoded_data, Label("CMV", [True, False]))
        self.assertEqual(500, len(predictions_proba["CMV"][True]))
        self.assertEqual(500, len(predictions_proba["CMV"][False]))
        self.assertEqual(500 * [1], list(predictions_proba["CMV"][True] + predictions_proba["CMV"][False]))

        cnn.store(path / "model_storage")

        cnn2 = ReceptorCNN(sequence_type="amino_acid")
        cnn2.load(path / "model_storage")

        cnn2_vars = vars(cnn2)
        del cnn2_vars["CNN"]
        cnn_vars = vars(cnn)
        del cnn_vars["CNN"]

        for item, value in cnn_vars.items():
            if isinstance(value, Label):
                self.assertDictEqual(vars(value), (vars(cnn2_vars[item])))
            elif not isinstance(value, np.ndarray):
                self.assertEqual(value, cnn2_vars[item])

        shutil.rmtree(path)
