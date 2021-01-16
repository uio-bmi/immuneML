import unittest
import numpy as np
import os
import shutil

from source.reports.ml_reports.ROCCurve import ROCCurve
from source.reports.ReportResult import ReportResult
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.LogisticRegression import LogisticRegression
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.dataset.Dataset import Dataset


class TestCoefficients(unittest.TestCase):

    def _create_dummy_lr_model(self, path):
        dummy_lr = LogisticRegression()
        dummy_lr.fit_by_cross_validation(EncodedData(np.random.rand(100, 20), {"l1": [i % 2 for i in range(0, 100)]}),
                                         number_of_splits=2, label_name="l1")
        return dummy_lr

    def _create_report(self, path):
        report = ROCCurve.build_object(name='testcase')

        report.method = self._create_dummy_lr_model(path)
        report.label = "l1"
        report.result_path = path
        report.test_dataset = Dataset()
        report.test_dataset.encoded_data = EncodedData(np.random.rand(100, 20), {"l1": [i % 2 for i in range(0, 100)]})

        return report

    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/roccurve/"

        report = self._create_report(path)

        # Running the report
        preq = report.check_prerequisites()
        result = report.generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(preq, True)
        self.assertEqual(os.path.isfile(f"{path}{'testcase'}.csv"), True)
        self.assertEqual(os.path.isfile(f"{path}{'testcase'}.html"), True)

        shutil.rmtree(path)

if __name__ == '__main__':
    unittest.main()
