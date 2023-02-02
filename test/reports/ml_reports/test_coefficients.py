import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
import yaml

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.CoefficientPlottingSetting import CoefficientPlottingSetting
from immuneML.reports.ml_reports.Coefficients import Coefficients
from immuneML.util.PathBuilder import PathBuilder


class TestCoefficients(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_lr_model(self, path):
        # dummy logistic regression with 100 observations with 20 features belonging to 2 classes
        dummy_lr = LogisticRegression()
        dummy_lr.fit_by_cross_validation(EncodedData(np.random.rand(100, 20), {"l1": [i % 2 for i in range(0, 100)]}),
                                         number_of_splits=2, label=Label("l1", [0, 1]))

        # Change coefficients to values 1-20
        dummy_lr.model.coef_ = np.array(list(range(0, 20))).reshape(1, -1)

        file_path = path / "ml_details.yaml"
        with file_path.open("w") as file:
            yaml.dump({"l1": {"feature_names": [f"feature{i}" for i in range(20)]}},
                      file)

        return dummy_lr

    def _create_report(self, path):
        report = Coefficients.build_object(**{"coefs_to_plot": [CoefficientPlottingSetting.ALL.name,
                                                                CoefficientPlottingSetting.NONZERO.name,
                                                                CoefficientPlottingSetting.CUTOFF.name,
                                                                CoefficientPlottingSetting.N_LARGEST.name],
                                              "cutoff": [10],
                                              "n_largest": [5]})

        report.method = self._create_dummy_lr_model(path)
        report.ml_details_path = path / "ml_details.yaml"
        report.label = Label("l1", [0, 1])
        report.result_path = path
        report.train_dataset = Dataset()
        report.train_dataset.encoded_data = EncodedData(examples=np.zeros((1, 20)), labels={"A": [1]},
                                                        feature_names=[f"feature{i}" for i in range(20)])

        return report

    def test_generate(self):
        path = EnvironmentSettings.root_path / "test/tmp/logregcoefsreport/"
        PathBuilder.build(path)

        report = self._create_report(path)

        # Running the report
        result = report.generate_report()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_tables[0].path, path / "coefficients.csv")
        self.assertEqual(result.output_figures[0].path, path / "all_coefficients.html")
        self.assertEqual(result.output_figures[1].path, path / "nonzero_coefficients.html")
        self.assertEqual(result.output_figures[2].path, path / "cutoff_10_coefficients.html")
        self.assertEqual(result.output_figures[3].path, path / "largest_5_coefficients.html")

        # Actual tests
        self.assertTrue(os.path.isfile(path / "coefficients.csv"))
        self.assertTrue(os.path.isfile(path / "all_coefficients.html"))
        self.assertTrue(os.path.isfile(path / "nonzero_coefficients.html"))
        self.assertTrue(os.path.isfile(path / "cutoff_10_coefficients.html"))
        self.assertTrue(os.path.isfile(path / "largest_5_coefficients.html"))

        written_data = pd.read_csv(path / "coefficients.csv")

        self.assertListEqual(list(written_data.columns), ["features", "coefficients"])
        self.assertListEqual(list(written_data["coefficients"]), list(reversed([i for i in range(20)])))
        self.assertListEqual(list(written_data["features"]), list(reversed([f"feature{i}" for i in range(20)])))

        shutil.rmtree(path)
