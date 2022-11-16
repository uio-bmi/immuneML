import os
import random
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.MotifClassifier import MotifClassifier
from immuneML.util.PathBuilder import PathBuilder


class TestMotifClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def get_enc_data(self):
        enc_data = EncodedData(encoding=MotifEncoder.__name__,
                               example_ids=["1", "2", "3", "4", "5", "6", "7", "8"],
                               feature_names=["useless_rule", "rule1", "rule2", "rule3"],
                               examples=np.array([[False, True, False, False],
                                                  [True, True, False, False],
                                                  [False, False, True, True],
                                                  [True, False, True, True],
                                                  [False, False, False, True],
                                                  [True, False, False, True],
                                                  [False, False, False, False],
                                                  [True, False, False, False]]),
                               labels={"l1": [True, True, True, True, False, False, False, False]})

        label = Label("l1", positive_class=True)
        return enc_data, label

    def get_fitted_classifier(self, enc_data, label):
        motif_classifier = MotifClassifier(training_percentage=0.7,
                                           max_motifs=100,
                                           patience=10,
                                           min_delta=0)

        random.seed(1)
        motif_classifier.fit(encoded_data=enc_data, label=label,
                             optimization_metric="accuracy")
        random.seed(None)

        return motif_classifier


    def test_fit(self):
        enc_data, label = self.get_enc_data()
        motif_classifier = self.get_fitted_classifier(enc_data, label)

        predictions = motif_classifier.predict(enc_data, label)

        self.assertListEqual(list(predictions.keys()), ["l1"])
        self.assertListEqual(list(predictions["l1"]), [True, True, True, True, False, False, False, False])

        self.assertListEqual(sorted(motif_classifier.rule_tree_features), ["rule1", "rule2"])
        self.assertDictEqual(motif_classifier.class_mapping, {0: False, 1: True})

    def test_load_store(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "motif_classifier")

        enc_data, label = self.get_enc_data()
        motif_classifier = self.get_fitted_classifier(enc_data, label)

        motif_classifier.store(path / "model_storage")

        motif_classifier2 = MotifClassifier()
        motif_classifier2.load(path / "model_storage")

        motif_classifier2_vars = vars(motif_classifier2)
        cnn_vars = vars(motif_classifier)

        for item, value in cnn_vars.items():
            if isinstance(value, Label):
                self.assertDictEqual(vars(value), (vars(motif_classifier2_vars[item])))
            else:
                self.assertEqual(value, motif_classifier2_vars[item])

        predictions = motif_classifier.predict(enc_data, label)
        predictions2 = motif_classifier2.predict(enc_data, label)

        self.assertListEqual(list(predictions.keys()), ["l1"])
        self.assertListEqual(list(predictions["l1"]), [True, True, True, True, False, False, False, False])

        self.assertListEqual(list(predictions2.keys()), ["l1"])
        self.assertListEqual(list(predictions2["l1"]), [True, True, True, True, False, False, False, False])

        shutil.rmtree(path)

    def test_recursively_select_rules(self):
        motif_classifier = MotifClassifier(max_motifs = 100,
                                           min_delta = 0,
                                           patience = 10)
        motif_classifier.optimization_metric = "accuracy"
        motif_classifier.class_mapping = {0: False, 1: True}
        motif_classifier.label = Label("l1", positive_class=True)
        motif_classifier.feature_names = ["rule"]

        enc_data = EncodedData(encoding=MotifEncoder.__name__,
                               example_ids=["1", "2", "3", "4"],
                               feature_names=["rule"],
                               examples=np.array([[False],
                                                  [True],
                                                  [False],
                                                  [True]]),
                               labels={"l1": [False, True, False, True]})

        result_no_improvement_on_training = motif_classifier._recursively_select_rules(enc_data, None, [1], [0])
        self.assertListEqual(result_no_improvement_on_training, [0])

        enc_data = EncodedData(encoding=MotifEncoder.__name__,
                               example_ids=["1", "2", "3", "4"],
                               feature_names=["rule1", "rule2", "rule3"],
                               examples=np.array([[True, False, False],
                                              [False, True, False],
                                              [False, False, False],
                                              [False, False, False]]),
                               labels={"l1": [True, True, True, True]})

        motif_classifier.feature_names = ["rule1", "rule2", "rule3"]

        result_add_one_rule = motif_classifier._recursively_select_rules(enc_data, enc_data, [0], [0])
        self.assertListEqual(result_add_one_rule, [0, 1])

        motif_classifier.max_motifs = 1

        result_max_motifs_reached = motif_classifier._recursively_select_rules(enc_data, enc_data, [], [])
        self.assertListEqual(result_max_motifs_reached, [0])

        motif_classifier.max_motifs = 2

        result_max_motifs_reached = motif_classifier._recursively_select_rules(enc_data, enc_data, [], [])
        self.assertListEqual(result_max_motifs_reached, [0, 1])


    def test_get_rule_tree_features_from_indices(self):
        motif_classifier = MotifClassifier()
        features = motif_classifier._get_rule_tree_features_from_indices([0, 2], ["A", "B", "C"])

        self.assertListEqual(features, ["A", "C"])

    def test_test_is_improvement(self):
        motif_classifier = MotifClassifier()

        result = motif_classifier._test_is_improvement([0.0, 0.1, 0.5, 1], 0.1)
        self.assertListEqual(result, [True, False, True, True])

        result = motif_classifier._test_is_improvement([0, 0, 0, 1], 0)
        self.assertListEqual(result, [True, False, False, True])

        result = motif_classifier._test_is_improvement([0], 0)
        self.assertListEqual(result, [True])

    def test_test_earlystopping(self):
        motif_classifier = MotifClassifier(patience=5)

        self.assertEqual(motif_classifier._test_earlystopping([]), False)
        self.assertEqual(motif_classifier._test_earlystopping([False, False, False]), False)
        self.assertEqual(motif_classifier._test_earlystopping([True, True, True]), False)
        self.assertEqual(motif_classifier._test_earlystopping([True, True, True, True, True]), False)
        self.assertEqual(motif_classifier._test_earlystopping([False, False, False, False, False]), True)
        self.assertEqual(motif_classifier._test_earlystopping([True, True, True, False, False, False, False, False]), True)

    def test_get_optimal_indices(self):
        motif_classifier = MotifClassifier(patience=3)


        result = motif_classifier._get_optimal_indices([1,2,3,4,5,6,7,8,9,10], [True, True, True, False, False])
        self.assertListEqual(result, [1,2,3])

        result = motif_classifier._get_optimal_indices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [True])
        self.assertListEqual(result, [1])

        result = motif_classifier._get_optimal_indices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [True, False, False, False, True])
        self.assertListEqual(result, [1, 2, 3, 4, 5])

    def test_get_rule_tree_predictions(self):
        enc_data = EncodedData(encoding=MotifEncoder.__name__,
                               example_ids=["1", "2", "3", "4"],
                               feature_names=["rule1", "rule2", "rule3"],
                               examples=np.array([[True, False, False],
                                                  [False, True, False],
                                                  [False, False, False],
                                                  [False, False, False]]),
                               labels={"l1": [True, True, True, True]})

        motif_classifier = MotifClassifier()
        motif_classifier.feature_names = ["rule1", "rule2", "rule3"]

        result = motif_classifier._get_rule_tree_predictions(enc_data, [0])
        self.assertListEqual(list(result), [True, False, False, False])

        result = motif_classifier._get_rule_tree_predictions(enc_data, [0, 1])
        self.assertListEqual(list(result), [True, True, False, False])

        result = motif_classifier._get_rule_tree_predictions(enc_data, [0, 1, 2])
        self.assertListEqual(list(result), [True, True, False, False])


