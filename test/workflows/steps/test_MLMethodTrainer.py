import shutil
from unittest import TestCase

import numpy as np

from source.data_model.dataset.Dataset import Dataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.LogisticRegression import LogisticRegression
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer


class TestMLMethodTrainer(TestCase):
    def test_check_prerequisites(self):
        self.assertRaises(AssertionError, MLMethodTrainer.check_prerequisites, {})
        self.assertRaises(AssertionError, MLMethodTrainer.check_prerequisites, None)
        self.assertRaises(AssertionError, MLMethodTrainer.check_prerequisites, {"dataset": None})
        self.assertRaises(AssertionError, MLMethodTrainer.check_prerequisites, {"labels": []})
        self.assertRaises(AssertionError, MLMethodTrainer.check_prerequisites, {"result_path": ""})

    def test_perform_step(self):
        method = LogisticRegression()
        dataset = Dataset()
        dataset.encoded_data = {
            "repertoires": np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4]]),
            "labels": np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]),
            "label_names": ["l1", "l2"]
        }

        path = EnvironmentSettings.root_path + "test/tmp/mlmethodtrainer/"

        method = MLMethodTrainer.perform_step({
            "result_path": path,
            "dataset": dataset,
            "labels": ["l1"],
            "method": method,
            "number_of_splits": 2
        })

        method.predict(np.array([1, 2, 3]).reshape(1, -1), ["l1"])

        shutil.rmtree(path)
