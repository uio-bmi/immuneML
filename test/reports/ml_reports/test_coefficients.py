import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.CoefficientPlottingSetting import CoefficientPlottingSetting
from immuneML.reports.ml_reports.Coefficients import Coefficients
from immuneML.util.PathBuilder import PathBuilder


class TestCoefficients(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_lr_model(self, path):
        dummy_lr = LogisticRegression()
        dummy_lr.fit_by_cross_validation(EncodedData(np.random.rand(100, 20), {"l1": [i % 2 for i in range(0, 100)]}),
                                         number_of_splits=2, label=Label("l1", [0, 1]), optimization_metric="balanced_accuracy")
        dummy_lr.model.coef_ = np.array(list(range(0, 20))).reshape(1, -1)
        return dummy_lr

    def _create_dummy_logreg_custom_model(self):
        from immuneML.ml_methods.classifiers.LogRegressionCustomPenalty import LogRegressionCustomPenalty

        model = LogRegressionCustomPenalty(backend='torch', n_lambda=5, n_splits=2,
                                           alpha=0.5, max_iter=100, device='cpu')
        model.label = Label("l1", [0, 1])
        model.class_mapping = {0: 0, 1: 1}
        encoded_data = EncodedData(
            examples=np.random.rand(60, 20).astype(np.float32),
            labels={"l1": np.array([i % 2 for i in range(60)])},
            feature_names=[f"feature{i}" for i in range(20)]
        )
        model._fit(encoded_data)
        return model

    def _create_dummy_xgb_model(self):
        from immuneML.ml_methods.classifiers.XGBClassifier import XGBClassifier

        model = XGBClassifier(n_estimators=10, verbosity=0)
        model.label = Label("l1", [0, 1])
        model.class_mapping = {0: 0, 1: 1}
        encoded_data = EncodedData(
            examples=np.random.rand(60, 20),
            labels={"l1": np.array([i % 2 for i in range(60)])},
            feature_names=[f"feature{i}" for i in range(20)]
        )
        model._fit(encoded_data)
        return model

    def _run_report_test(self, path, method):
        report = Coefficients.build_object(**{"coefs_to_plot": [CoefficientPlottingSetting.ALL.name,
                                                                CoefficientPlottingSetting.NONZERO.name,
                                                                CoefficientPlottingSetting.CUTOFF.name,
                                                                CoefficientPlottingSetting.N_LARGEST.name],
                                              "cutoff": [10],
                                              "n_largest": [5]})
        report.method = method
        report.label = Label("l1", [0, 1])
        report.result_path = path
        report.train_dataset = Dataset()
        report.train_dataset.encoded_data = EncodedData(examples=np.zeros((1, 20)), labels={"A": [1]},
                                                        feature_names=[f"feature{i}" for i in range(20)])

        self.assertTrue(report.check_prerequisites())
        result = report._generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_tables[0].path, path / "coefficients.csv")

        self.assertTrue(os.path.isfile(path / "coefficients.csv"))
        self.assertTrue(os.path.isfile(path / "all_coefficients.html"))
        self.assertTrue(os.path.isfile(path / "nonzero_coefficients.html"))
        self.assertTrue(os.path.isfile(path / "largest_5_coefficients.html"))

        written_data = pd.read_csv(path / "coefficients.csv")
        self.assertListEqual(list(written_data.columns), ["features", "coefficients"])
        self.assertEqual(len(written_data), 20)
        return written_data

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "coefs_report/"
        PathBuilder.remove_old_and_build(path)

        lr_path = path / "logreg"
        PathBuilder.build(lr_path)
        written_data = self._run_report_test(lr_path, self._create_dummy_lr_model(lr_path))
        # LR coefficients are set to 0-19, so all four plot variants are generated (cutoff 10 is non-empty)
        self.assertTrue(os.path.isfile(lr_path / "cutoff_10_coefficients.html"))
        self.assertListEqual(list(written_data["coefficients"]), list(reversed([i for i in range(20)])))
        self.assertListEqual(list(written_data["features"]), list(reversed([f"feature{i}" for i in range(20)])))

        logreg_custom_path = path / "logreg_custom"
        PathBuilder.build(logreg_custom_path)
        written_data = self._run_report_test(logreg_custom_path, self._create_dummy_logreg_custom_model())
        self.assertTrue(all(isinstance(v, float) for v in written_data["coefficients"]))

        xgb_path = path / "xgb"
        PathBuilder.build(xgb_path)
        written_data = self._run_report_test(xgb_path, self._create_dummy_xgb_model())
        self.assertTrue(all(v >= 0 for v in written_data["coefficients"]))  # importances are non-negative

        shutil.rmtree(path)