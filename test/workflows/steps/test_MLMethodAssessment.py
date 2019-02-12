from unittest import TestCase

import numpy as np

from source.data_model.dataset.Dataset import Dataset
from source.environment.MetricType import MetricType
from source.ml_methods.LogisticRegression import LogisticRegression
from source.ml_methods.SVM import SVM
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment


class TestMLMethodAssessment(TestCase):

    def test_perform_step(self):
        dataset = Dataset()
        dataset.encoded_data = {
            "repertoires": np.array([[1,2], [1,2], [1,2], [1,2], [1,2], [1,2]]),
            "labels": np.array([[1,2,3,1,2,3], [1,2,3,1,2,3]]),
            "label_names": ["l1", "l2"]
        }

        method1 = SVM()
        method1.fit(dataset.encoded_data["repertoires"], dataset.encoded_data["labels"], dataset.encoded_data["label_names"])

        method2 = LogisticRegression()
        method2.fit(dataset.encoded_data["repertoires"], dataset.encoded_data["labels"],
                    dataset.encoded_data["label_names"])

        res = MLMethodAssessment.perform_step({
            "dataset": dataset,
            "methods": [method1, method2],
            "labels": ["l1", "l2"],
            "metrics": [MetricType.ACCURACY, MetricType.BALANCED_ACCURACY, MetricType.F1_MACRO],
            "predictions_path": None
        })

        self.assertTrue("SVM" in res)
        self.assertTrue("LogisticRegression" in res)
        self.assertTrue("l1" in res["SVM"] and "l2" in res["LogisticRegression"])
        self.assertTrue("ACCURACY" in res["SVM"]["l1"] and "F1_MACRO" in res["LogisticRegression"]["l2"])
