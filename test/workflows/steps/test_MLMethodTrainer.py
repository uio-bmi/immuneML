import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.workflows.steps.MLMethodTrainer import MLMethodTrainer
from immuneML.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams


class TestMLMethodTrainer(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        method = LogisticRegression()
        dataset = RepertoireDataset()
        dataset.encoded_data = EncodedData(
            examples=np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4]]),
            labels={
                "l1": [1, 0, 1, 0, 1, 0],
                "l2": [0, 1, 0, 1, 0, 1]
            },
            feature_names=["f1", "f2", "f3"]
        )

        path = EnvironmentSettings.root_path / "test/tmp/mlmethodtrainer/"

        method = MLMethodTrainer.run(MLMethodTrainerParams(
            result_path=path,
            dataset=dataset,
            label=Label(name="l1", values=[0,1]),
            method=method,
            model_selection_n_folds=2,
            model_selection_cv=True,
            cores_for_training=1,
            train_predictions_path=path / "predictions.csv",
            ml_details_path=path / "details.yaml",
            optimization_metric="balanced_accuracy"
        ))

        method.predict(EncodedData(np.array([1, 2, 3]).reshape(1, -1)), Label("l1", [0, 1]))
        self.assertTrue(os.path.isfile(path / "predictions.csv"))
        self.assertTrue(os.path.isfile(path / "details.yaml"))

        shutil.rmtree(path)
