import os
import shutil

import numpy as np
import pandas as pd
import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.ConfusionMatrix import ConfusionMatrix
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


@pytest.fixture(autouse=True)
def set_test_cache():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name


@pytest.fixture
def dummy_test_data():
    np.random.seed(0)
    features = np.random.rand(20, 5)
    labels = [i % 3 for i in range(20)]  # 3 classes: 0, 1, 2

    return EncodedData(examples=features, labels={"label1": labels})


@pytest.fixture
def trained_model(dummy_test_data):
    model = LogisticRegression()
    model.fit(dummy_test_data, Label("label1", [0, 1, 2]))
    return model


def test_confusion_matrix_report(dummy_test_data, trained_model):
    result_path = EnvironmentSettings.tmp_test_path / "confusion_matrix_report"
    PathBuilder.remove_old_and_build(result_path)

    report = ConfusionMatrix.build_object(name="confmat_report", method=trained_model,
                                          train_dataset=SequenceDataset(encoded_data=dummy_test_data),
                                          test_dataset=SequenceDataset(encoded_data=dummy_test_data),
                                          result_path=result_path, label=Label("label1", [0, 1, 2]))

    assert report.check_prerequisites()

    result: ReportResult = report._generate()

    # Check ReportResult contents
    assert isinstance(result, ReportResult)
    assert result.output_tables[0].path.name == "confusion_matrix.csv"
    assert result.output_figures[0].path.name == "confusion_matrix.html"

    # Check files exist
    assert (result_path / "confusion_matrix.csv").is_file()
    assert (result_path / "confusion_matrix.html").is_file()

    # Check CSV content
    cm_df = pd.read_csv(result_path / "confusion_matrix.csv", index_col=0)
    assert cm_df.shape == (3, 3)  # 3 classes
    assert list(cm_df.columns) == ["0", "1", "2"]
    assert list(cm_df.index) == [0, 1, 2]

    # Clean up
    shutil.rmtree(result_path)


def test_confusion_matrix_report_alternative_label(dummy_test_data, trained_model):
    result_path = EnvironmentSettings.tmp_test_path / "confusion_matrix_report_alt_label"
    PathBuilder.remove_old_and_build(result_path)

    test_dataset = RandomDatasetGenerator.generate_sequence_dataset(20, {5: 1,},
                                                                    {'alt_label': {'A': 0.5, 'B': 0.33, 'C': 0.17}}, result_path)
    test_dataset.encoded_data = dummy_test_data

    report = ConfusionMatrix.build_object(name="confmat_report_alt", method=trained_model,
                                          train_dataset=SequenceDataset(encoded_data=dummy_test_data),
                                          test_dataset=test_dataset,
                                          result_path=result_path, label=Label("label1", [0, 1, 2]),
                                          alternative_label='alt_label')

    assert report.check_prerequisites()

    result: ReportResult = report._generate()

    # Check ReportResult contents
    assert isinstance(result, ReportResult)
    assert result.output_tables[0].path.name == "confusion_matrix.csv"
    assert len(result.output_figures) == 2

    # Check files exist
    assert (result_path / "confusion_matrix.csv").is_file()

    # Clean up
    shutil.rmtree(result_path)
