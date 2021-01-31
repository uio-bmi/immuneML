import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
import yaml

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset  import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.ConfounderAnalysis import ConfounderAnalysis
from immuneML.util.PathBuilder import PathBuilder


class TestConfounderAnalysis(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_lr_model(self, path, encoded_data):
        # dummy logistic regression with 100 observations with 3 features belonging to 2 classes
        dummy_lr = LogisticRegression()
        dummy_lr.fit_by_cross_validation(encoded_data,
                                         number_of_splits=2, label_name="signal_disease")

        file_path = path / "ml_details.yaml"
        with file_path.open("w") as file:
            yaml.dump({"signal_disease": {"feature_names": ["feat1", "feat2", "signal_age"]}},
                      file)

        return dummy_lr

    def _create_report(self, path):
        # todo add HLA
        report = ConfounderAnalysis.build_object(**{"additional_labels": ["signal_age"]})

        encoded_data = EncodedData(examples=np.hstack((np.random.randn(100,2), np.random.choice([0, 1], size=(100,1), p=[1. / 3, 2. / 3]))),
                                   labels={"signal_disease": list(np.random.choice([0, 1], size=(100,), p=[1. / 3, 2. / 3]))},
                                   feature_names=["feat1", "feat2", "signal_age"])

        report.method = self._create_dummy_lr_model(path, encoded_data)
        report.ml_details_path = path / "ml_details.yaml"
        report.label = "signal_disease"
        report.result_path = path
        report.train_dataset = RepertoireDataset()
        report.train_dataset.encoded_data = encoded_data
        report.test_dataset = RepertoireDataset()
        report.test_dataset.encoded_data = encoded_data

        return report

    def test_generate(self):
        path = EnvironmentSettings.root_path / "test/tmp/logregconfreport/"
        PathBuilder.build(path)

        report = self._create_report(path)

        # Running the report
        result = report.generate_report()
        return report
