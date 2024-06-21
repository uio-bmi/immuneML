import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFreqSequenceEncoder import KmerFreqSequenceEncoder
from immuneML.encodings.onehot.OneHotReceptorEncoder import OneHotReceptorEncoder
from immuneML.encodings.onehot.OneHotSequenceEncoder import OneHotSequenceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.ml_methods.classifiers.KerasSequenceCNN import KerasSequenceCNN
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestKerasSequenceCNN(TestCase):
    maxDiff = None

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_if_keras_installed(self):
        try:
            import keras
            from keras.optimizers import Adam
            self._test_fit()
        except ImportError as e:
            print("Test ignored since keras is not installed.")

    def _recursive_convert_lists_to_tuples(self, model_description):
        if isinstance(model_description, dict):
            return {key: self._recursive_convert_lists_to_tuples(value) for key, value in
                    model_description.items()}
        elif isinstance(model_description, (list, tuple, set)):
            return tuple([self._recursive_convert_lists_to_tuples(item) for item in model_description])
        else:
            return model_description

    def _test_fit(self):
        import keras
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "keras_cnn")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=500, length_probabilities={5: 1},
                                                                   labels={"CMV": {"yes": 0.5, "no": 0.5}}, path=path / "dataset")

        label = Label("CMV", values=["yes", "no"], positive_class="yes")
        encoder = OneHotSequenceEncoder(False, None, False, "enc1")
        enc_dataset = encoder.encode(dataset, EncoderParams(path / "result", LabelConfiguration([label])))

        cnn = KerasSequenceCNN(units_per_layer=[['CONV', 400, 3, 1],
                                                  ['DROP', 0.5],
                                                  ['POOL', 2, 1],
                                                  ['FLAT'],
                                                  ['DENSE', 50]],
                               activation="relu",
                               training_percentage=0.7)

        cnn.check_encoder_compatibility(encoder)
        self.assertRaises(ValueError, lambda: cnn.check_encoder_compatibility(OneHotReceptorEncoder(use_positional_info=False, distance_to_seq_middle=1, flatten=False)))
        self.assertRaises(ValueError, lambda: cnn.check_encoder_compatibility(KmerFreqSequenceEncoder(normalization_type=None, reads=None, sequence_encoding=None)))
        self.assertRaises(AssertionError, lambda: cnn.check_encoder_compatibility(OneHotSequenceEncoder(use_positional_info=True, distance_to_seq_middle=1, flatten=False)))
        self.assertRaises(AssertionError, lambda: cnn.check_encoder_compatibility(OneHotSequenceEncoder(use_positional_info=False, distance_to_seq_middle=1, flatten=True)))

        cnn.fit(encoded_data=enc_dataset.encoded_data, label=label)

        predictions = cnn.predict(enc_dataset.encoded_data, label)
        self.assertEqual(500, len(predictions["CMV"]))
        self.assertEqual(500, len([pred for pred in predictions["CMV"]]))

        predictions_proba = cnn.predict_proba(enc_dataset.encoded_data, label)
        self.assertEqual(500 * [1], list(predictions_proba["CMV"]["yes"] + predictions_proba["CMV"]["no"]))
        self.assertEqual(500, predictions_proba["CMV"]["yes"].shape[0])
        self.assertEqual(500, predictions_proba["CMV"]["no"].shape[0])

        self.assertListEqual(list(predictions_proba["CMV"]["yes"] > 0.5), [pred == "yes" for pred in list(predictions["CMV"])])

        cnn.store(path / "model_storage")

        cnn2 = KerasSequenceCNN()
        cnn2.load(path / "model_storage")

        cnn2_params = cnn2.get_params()
        cnn_params = cnn.get_params()

        for item, value in cnn_params.items():
            if isinstance(value, Label):
                self.assertDictEqual(vars(value), (vars(cnn2_params[item])))
            elif item == "model":
                model1_params = self._recursive_convert_lists_to_tuples(cnn_params["model"])
                model2_params = self._recursive_convert_lists_to_tuples(cnn2_params["model"])

                self.assertDictEqual(model1_params, model2_params)
            else:
                self.assertEqual(value, cnn2_params[item])

        predictions_proba2 = cnn2.predict_proba(enc_dataset.encoded_data, label)

        self.assertTrue(all(predictions_proba["CMV"]["yes"] == predictions_proba2["CMV"]["yes"]))
        self.assertTrue(all(predictions_proba["CMV"]["no"] == predictions_proba2["CMV"]["no"]))

        shutil.rmtree(path)


