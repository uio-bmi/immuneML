import os
import shutil
import unittest

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.ROCCurve import ROCCurve


class TestROCCurve(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_lr_model(self):
        dummy_lr = LogisticRegression()
        encoded_tr = EncodedData(np.random.rand(100, 20),
                                 {"l1": [i % 2 for i in range(0, 100)]})

        dummy_lr.fit_by_cross_validation(encoded_tr, number_of_splits=2, optimization_metric="balanced_accuracy",
                                         label=Label("l1", [0, 1], positive_class=1))
        return dummy_lr

    def _create_report(self, path):
        report = ROCCurve.build_object(name='testcase')

        report.method = self._create_dummy_lr_model()
        report.label = Label("l1", [0, 1], positive_class=1)
        report.result_path = path
        report.test_dataset = Dataset()
        encoded_te = EncodedData(np.random.rand(100, 20),
                                 {"l1": [i % 2 for i in range(0, 100)]})

        report.test_dataset.encoded_data = encoded_te

        return report

    def test_generate(self):
        path = EnvironmentSettings.root_path / "test/tmp/roccurve/"

        report = self._create_report(path)

        # Running the report
        preq = report.check_prerequisites()
        self.assertTrue(preq)
        result = report._generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(preq, True)
        self.assertEqual(os.path.isfile(path / "testcase.csv"), True)
        self.assertEqual(os.path.isfile(path / "testcase.html"), True)

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
