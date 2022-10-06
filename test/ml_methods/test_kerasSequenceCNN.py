import os
import shutil
from unittest import TestCase
import numpy as np
import keras

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFreqSequenceEncoder import KmerFreqSequenceEncoder
from immuneML.encodings.onehot.OneHotReceptorEncoder import OneHotReceptorEncoder
from immuneML.encodings.onehot.OneHotSequenceEncoder import OneHotSequenceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.ml_methods.KerasSequenceCNN import KerasSequenceCNN
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder




class TestKerasSequenceCNN(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_fit(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "keras_cnn")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=500, length_probabilities={5: 1},
                                                                   labels={"CMV": {True: 0.5, False: 0.5}}, path=path / "dataset")

        encoder = OneHotSequenceEncoder(False, None, False, "enc1")
        enc_dataset = encoder.encode(dataset, EncoderParams(path / "result",
                                                                                        LabelConfiguration([Label("CMV", [True, False])])))

        cnn = KerasSequenceCNN(units_per_layer=[['CONV', 400, 3, 1],
                                                  ['DROP', 0.5],
                                                  ['POOL', 2, 1],
                                                  ['FLAT'],
                                                  ['DENSE', 50]],
                               activation="relu",
                               regularizer=None,
                               training_percentage=0.7)

        cnn.check_encoder_compatibility(encoder)
        self.assertRaises(ValueError, lambda: cnn.check_encoder_compatibility(OneHotReceptorEncoder(use_positional_info=False, distance_to_seq_middle=1, flatten=False)))
        self.assertRaises(ValueError, lambda: cnn.check_encoder_compatibility(KmerFreqSequenceEncoder(normalization_type=None, reads=None, sequence_encoding=None)))
        self.assertRaises(AssertionError, lambda: cnn.check_encoder_compatibility(OneHotSequenceEncoder(use_positional_info=True, distance_to_seq_middle=1, flatten=False)))
        self.assertRaises(AssertionError, lambda: cnn.check_encoder_compatibility(OneHotSequenceEncoder(use_positional_info=False, distance_to_seq_middle=1, flatten=True)))


        cnn.fit(encoded_data=enc_dataset.encoded_data, label=Label("CMV"))

        predictions_proba = cnn.predict_proba(enc_dataset.encoded_data, Label("CMV"))
        self.assertEqual(500, np.rint(np.sum(predictions_proba["CMV"])))
        self.assertEqual(500, predictions_proba["CMV"].shape[0])
        self.assertEqual(2, predictions_proba["CMV"].shape[1])

        predictions = cnn.predict(enc_dataset.encoded_data, Label("CMV"))
        self.assertEqual(500, len(predictions["CMV"]))
        self.assertEqual(500, len([pred for pred in predictions["CMV"] if isinstance(pred, bool)]))

        self.assertListEqual(list(predictions_proba["CMV"][:, 1] > 0.5), list(predictions["CMV"]))

        cnn.store(path / "model_storage")

        cnn2 = KerasSequenceCNN()
        cnn2.load(path / "model_storage")

        cnn2_vars = vars(cnn2)
        del cnn2_vars["CNN"]
        cnn_vars = vars(cnn)
        del cnn_vars["CNN"]

        for item, value in cnn_vars.items():
            if isinstance(value, Label):
                self.assertDictEqual(vars(value), (vars(cnn2_vars[item])))
            elif not isinstance(value, keras.Sequential):
                self.assertEqual(value, cnn2_vars[item])

        predictions_proba2 = cnn2.predict_proba(enc_dataset.encoded_data, Label("CMV"))

        print(predictions_proba2)

        for i in range(len(predictions_proba["CMV"])):
            self.assertTrue(all(predictions_proba["CMV"][i] == predictions_proba2["CMV"][i]))

        shutil.rmtree(path)


