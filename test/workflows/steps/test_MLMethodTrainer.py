import shutil
from unittest import TestCase

import numpy as np

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.LogisticRegression import LogisticRegression
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer
from source.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams


class TestMLMethodTrainer(TestCase):

    def test_run(self):
        method = LogisticRegression()
        dataset = Dataset()
        dataset.encoded_data = EncodedData(
            repertoires=np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4]]),
            labels={
                "l1": [1, 0, 1, 0, 1, 0],
                "l2": [0, 1, 0, 1, 0, 1]
            },
            feature_names=["f1", "f2", "f3"]
        )

        path = EnvironmentSettings.root_path + "test/tmp/mlmethodtrainer/"

        method = MLMethodTrainer.run(MLMethodTrainerParams(
            result_path=path,
            dataset=dataset,
            labels=["l1"],
            method=method,
            model_selection_n_folds=2,
            model_selection_cv=True,
            cores_for_training=1
        ))

        method.predict(np.array([1, 2, 3]).reshape(1, -1), ["l1"])

        shutil.rmtree(path)
