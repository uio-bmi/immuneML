import os
import shutil
from unittest import TestCase

import numpy as np
import random

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.BinaryFeatureClassifier import BinaryFeatureClassifier
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.BinaryFeaturePrecisionRecall import BinaryFeaturePrecisionRecall
from immuneML.util.PathBuilder import PathBuilder



class TestBinaryFeaturePrecisionRecall(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_report(self, path, keep_all):
        enc_data_train = EncodedData(encoding=MotifEncoder.__name__,
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
                               labels={"l1": ["yes", "yes", "yes", "yes", "no", "no", "no", "no"]})

        enc_data_test = EncodedData(encoding=MotifEncoder.__name__,
                                    example_ids=["9", "10"],
                                    feature_names=["useless_rule", "rule1", "rule2", "rule3"],
                                    examples=np.array([[True, False, False, False],
                                                       [True, False, True, False]]),
                                    labels={"l1": ["yes", "no"]})

        label = Label("l1", values=["yes", "no"], positive_class="yes")

        motif_classifier = BinaryFeatureClassifier(training_percentage=0.7,
                                                   random_seed=1,
                                                   max_features=100,
                                                   patience=10,
                                                   min_delta=0,
                                                   keep_all=keep_all,
                                                   result_path=path)

        random.seed(1)
        motif_classifier.fit(encoded_data=enc_data_train, label=label,
                             optimization_metric="accuracy")
        random.seed(None)

        report = BinaryFeaturePrecisionRecall.build_object(**{})

        report.method = motif_classifier
        report.label = label
        report.result_path = path
        report.train_dataset = SequenceDataset(buffer_type="NA", dataset_file="", batchfiles_path="")
        report.test_dataset = SequenceDataset(buffer_type="NA", dataset_file="", batchfiles_path="")
        report.train_dataset.encoded_data = enc_data_train
        report.test_dataset.encoded_data = enc_data_test

        return report


    def test_generate_keep_all_false(self):
        path = EnvironmentSettings.root_path / "test/tmp/binary_feature_precision_recall"
        PathBuilder.build(path)

        report = self._create_report(path, keep_all=False)

        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "training_performance.tsv"))
        self.assertTrue(os.path.isfile(path / "validation_performance.tsv"))
        self.assertTrue(os.path.isfile(path / "test_performance.tsv"))
        self.assertTrue(os.path.isfile(path / "training_precision_recall.html"))
        self.assertTrue(os.path.isfile(path / "validation_precision_recall.html"))
        self.assertTrue(os.path.isfile(path / "test_precision_recall.html"))

        shutil.rmtree(path)

    def test_generate_keep_all_true(self):
        path = EnvironmentSettings.root_path / "test/tmp/binary_feature_precision_recall_keep_all"
        PathBuilder.build(path)

        report = self._create_report(path, keep_all=True)

        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "training_performance.tsv"))
        self.assertTrue(os.path.isfile(path / "test_performance.tsv"))
        self.assertTrue(os.path.isfile(path / "training_precision_recall.html"))
        self.assertTrue(os.path.isfile(path / "test_precision_recall.html"))

        shutil.rmtree(path)



