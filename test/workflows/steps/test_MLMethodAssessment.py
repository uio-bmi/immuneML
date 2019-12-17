import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.ml_methods.SVM import SVM
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment
from source.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams


class TestMLMethodAssessment(TestCase):

    def test_run(self):
        path = EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/"
        PathBuilder.build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["AA"], ["CC"], ["AA"], ["CC"], ["AA"], ["CC"]], path)[0])
        dataset.encoded_data = EncodedData(
            examples=np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]),
            labels={"l1": [1, 2, 3, 1, 2, 3], "l2": [1, 2, 3, 1, 2, 3]}
        )

        label_config = LabelConfiguration()
        label_config.add_label("l1", [1, 2, 3])
        label_config.add_label("l2", [1, 2, 3])

        method1 = SVM()
        method1.fit(dataset.encoded_data.examples, dataset.encoded_data.labels, dataset.encoded_data.labels.keys())

        res = MLMethodAssessment.run(MLMethodAssessmentParams(
            dataset=dataset,
            method=method1,
            metrics={MetricType.ACCURACY, MetricType.BALANCED_ACCURACY, MetricType.F1_MACRO},
            predictions_path=EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/fold_predictions.csv",
            label_configuration=label_config,
            ml_details_path=EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/ml_details.csv",
            run=1,
            all_predictions_path=EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/predictions.csv",
            path=EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/"
        ))

        self.assertTrue("l1" in res)
        self.assertTrue("l2" in res)

        self.assertTrue(os.path.isfile(EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/ml_details.csv"))

        df = pd.read_csv(EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/ml_details.csv")
        self.assertTrue(df.shape[0] == 1)

        df = pd.read_csv(EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/predictions.csv")
        self.assertEqual(6, df.shape[0])

        shutil.rmtree(EnvironmentSettings.root_path + "test/tmp/mlmethodassessment/")
