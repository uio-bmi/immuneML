import os
import shutil
from unittest import TestCase

import numpy as np

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer
from source.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams


class TestMLMethodTrainer(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        method = SimpleLogisticRegression()
        dataset = RepertoireDataset()
        dataset.encoded_data = EncodedData(
            examples=np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4]]),
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
            label="l1",
            method=method,
            model_selection_n_folds=2,
            model_selection_cv=True,
            cores_for_training=1,
            train_predictions_path=f"{path}predictions.csv",
            ml_details_path=f"{path}details.yaml",
            optimization_metric="balanced_accuracy"
        ))

        method.predict(EncodedData(np.array([1, 2, 3]).reshape(1, -1)), "l1")
        self.assertTrue(os.path.isfile(f"{path}predictions.csv"))
        self.assertTrue(os.path.isfile(f"{path}details.yaml"))

        shutil.rmtree(path)
