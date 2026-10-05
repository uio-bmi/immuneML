import os
import shutil
import numpy as np
from unittest import TestCase

from immuneML.environment.Label import Label
from immuneML.caching.CacheType import CacheType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.environment.Constants import Constants
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.classifiers.SillyClassifier import SillyClassifier


class TestSillyClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def get_enc_data(self):
        # Creates a mock encoded data object with 8 random examples
        enc_data = EncodedData(examples=np.array([[1, 0, 0],
                                                  [0, 1, 1],
                                                  [1, 1, 1],
                                                  [0, 1, 1],
                                                  [1, 0, 0],
                                                  [0, 1, 1],
                                                  [1, 1, 1],
                                                  [0, 1, 1]]),
                               example_ids=list(range(8)),
                               feature_names=["a", "b", "c"],
                               labels={"my_label": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"]},
                               encoding="random")

        label = Label(name="my_label", values=["yes", "no"], positive_class="yes")

        return enc_data, label

    def test_predictions(self):
        enc_data, label = self.get_enc_data()
        classifier = SillyClassifier(random_seed=50)

        # test fitting
        classifier.fit(enc_data, label)
        self.assertTrue(classifier.silly_model_fitted)

        # test 'predict'
        predictions = classifier.predict(enc_data, label)
        self.assertEqual(len(predictions[label.name]), len(enc_data.examples))

        # test 'predict_proba'
        prediction_probabilities = classifier.predict_proba(enc_data, label)
        self.assertEqual(len(prediction_probabilities[label.name][label.positive_class]), len(enc_data.examples))
        self.assertEqual(len(prediction_probabilities[label.name][label.get_binary_negative_class()]), len(enc_data.examples))
        self.assertTrue(all(0 <= pred <= 1 for pred in prediction_probabilities[label.name][label.positive_class]))
        self.assertTrue(all(0 <= pred <= 1 for pred in prediction_probabilities[label.name][label.get_binary_negative_class()]))

    def test_store_and_load(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "silly")
        enc_data, label = self.get_enc_data()
        classifier = SillyClassifier(random_seed=50)
        classifier.fit(enc_data, label)
        classifier.store(path)

        # Loading should be done in an 'empty' model (no parameters)
        classifier2 = SillyClassifier()
        classifier2.load(path)

        self.assertEqual(classifier.get_params(), classifier2.get_params())
        shutil.rmtree(path)

