import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.ml_methods.SVM import SVM
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment


class TestMLMethodAssessment(TestCase):

    def test_perform_step(self):
        dataset = Dataset()
        dataset.encoded_data = {
            "repertoires": np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]),
            "labels": np.array([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]),
            "label_names": ["l1", "l2"]
        }

        label_config = LabelConfiguration()
        label_config.add_label("l1", [1, 2, 3])
        label_config.add_label("l2", [1, 2, 3])

        method1 = SVM()
        method1.fit(dataset.encoded_data["repertoires"], dataset.encoded_data["labels"], dataset.encoded_data["label_names"])

        res = MLMethodAssessment.perform_step({
            "dataset": dataset,
            "method": method1,
            "labels": ["l1", "l2"],
            "metrics": [MetricType.ACCURACY, MetricType.BALANCED_ACCURACY, MetricType.F1_MACRO],
            "predictions_path":  EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/predictions/",
            "label_configuration": label_config,
            "ml_details_path": EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/ml_details.csv",
            "run": 1
        })

        self.assertTrue("l1_accuracy" in res.keys())
        self.assertTrue("l2_accuracy" in res.keys())
        self.assertTrue("l1_f1_macro" in res.keys())

        self.assertTrue(os.path.isfile(EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/ml_details.csv"))

        df = pd.read_csv(EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/ml_details.csv")
        self.assertTrue(df.shape[0] == 1)

        shutil.rmtree(EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/")
