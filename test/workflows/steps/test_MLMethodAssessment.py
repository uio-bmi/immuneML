import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder
from immuneML.workflows.steps.MLMethodAssessment import MLMethodAssessment
from immuneML.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams


class TestMLMethodAssessment(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path / "mlmethodassessment/"
        PathBuilder.build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build(
            [["AA"], ["CC"], ["AA"], ["CC"], ["AA"], ["CC"], ["AA"], ["CC"], ["AA"], ["CC"], ["AA"], ["CC"]], path)[0])
        dataset.encoded_data = EncodedData(
            examples=np.array([[1, 1], [1, 1], [3, 3], [1, 1], [1, 1], [3, 3], [1, 1], [1, 1], [3, 3], [1, 1], [1, 1], [3, 3]]),
            labels={"l1": [1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3], "l2": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]}
        )

        label_config = LabelConfiguration()
        label_config.add_label("l1", [1, 3], positive_class=3)

        label = Label(name='l1', values=[1, 3], positive_class=3)

        method1 = LogisticRegression()
        method1.fit(dataset.encoded_data, label=label)

        res = MLMethodAssessment.run(MLMethodAssessmentParams(
            dataset=dataset,
            method=method1,
            metrics={ClassificationMetric.ACCURACY, ClassificationMetric.BALANCED_ACCURACY, ClassificationMetric.F1_MACRO},
            optimization_metric=ClassificationMetric.LOG_LOSS,
            predictions_path=EnvironmentSettings.tmp_test_path / "mlmethodassessment/predictions.csv",
            label=label,
            ml_score_path=EnvironmentSettings.tmp_test_path / "mlmethodassessment/ml_score.csv",
            split_index=1,
            path=EnvironmentSettings.tmp_test_path / "mlmethodassessment/"
        ))

        self.assertTrue(isinstance(res, dict))
        self.assertTrue(res[ClassificationMetric.LOG_LOSS.name.lower()] <= 0.1)

        self.assertTrue(os.path.isfile(EnvironmentSettings.tmp_test_path / "mlmethodassessment/ml_score.csv"))

        df = pd.read_csv(EnvironmentSettings.tmp_test_path / "mlmethodassessment/ml_score.csv")
        self.assertEqual(df.shape[0], 1)

        df = pd.read_csv(EnvironmentSettings.tmp_test_path / "mlmethodassessment/predictions.csv")
        self.assertEqual(12, df.shape[0])

        shutil.rmtree(EnvironmentSettings.tmp_test_path / "mlmethodassessment/")
