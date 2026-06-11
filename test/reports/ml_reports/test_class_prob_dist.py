import os
import shutil

import numpy as np
import pandas as pd
import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.ClassProbabilityDistribution import ClassProbabilityDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


@pytest.fixture(autouse=True)
def set_test_cache():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name


N_EXAMPLES = 30
N_FEATURES = 10


@pytest.fixture
def binary_encoded_data():
    np.random.seed(42)
    return EncodedData(
        examples=np.random.rand(N_EXAMPLES, N_FEATURES),
        labels={"label1": [i % 2 for i in range(N_EXAMPLES)]}
    )


@pytest.fixture
def multiclass_encoded_data():
    np.random.seed(42)
    return EncodedData(
        examples=np.random.rand(N_EXAMPLES, N_FEATURES),
        labels={"label1": [i % 3 for i in range(N_EXAMPLES)]}
    )


@pytest.fixture
def binary_model(binary_encoded_data):
    model = LogisticRegression()
    model.fit(binary_encoded_data, Label("label1", [0, 1], positive_class=1))
    return model


@pytest.fixture
def multiclass_model(multiclass_encoded_data):
    model = LogisticRegression()
    model.fit(multiclass_encoded_data, Label("label1", [0, 1, 2]))
    return model


def _make_report(result_path, model, train_encoded, test_encoded, label, stratify_by=None):
    train_ds = SequenceDataset(encoded_data=train_encoded)
    test_ds = SequenceDataset(encoded_data=test_encoded)
    return ClassProbabilityDistribution.build_object(
        name="prob_report",
        method=model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        result_path=result_path,
        label=label,
        stratify_by=stratify_by
    )


def test_binary_generates_train_and_test_outputs(binary_encoded_data, binary_model, tmp_path):
    result_path = tmp_path / "binary"
    PathBuilder.build(result_path)
    label = Label("label1", [0, 1], positive_class=1)

    report = _make_report(result_path, binary_model, binary_encoded_data, binary_encoded_data, label)

    assert report.check_prerequisites()

    result: ReportResult = report._generate()

    assert isinstance(result, ReportResult)
    assert len(result.output_figures) == 2
    assert len(result.output_tables) == 2

    figure_names = {f.name for f in result.output_figures}
    assert "Class probability distribution (train)" in figure_names
    assert "Class probability distribution (test)" in figure_names

    assert (result_path / "prob_report_train.html").is_file()
    assert (result_path / "prob_report_test.html").is_file()
    assert (result_path / "prob_report_train.csv").is_file()
    assert (result_path / "prob_report_test.csv").is_file()


def test_csv_has_expected_columns_and_rows(binary_encoded_data, binary_model, tmp_path):
    result_path = tmp_path / "csv_check"
    PathBuilder.build(result_path)
    label = Label("label1", [0, 1], positive_class=1)

    report = _make_report(result_path, binary_model, binary_encoded_data, binary_encoded_data, label)
    report._generate()

    df = pd.read_csv(result_path / "prob_report_test.csv")
    assert set(df.columns) == {"true_class", "probability_for_class", "probability"}
    # one row per example per class: N_EXAMPLES * 2 classes
    assert len(df) == N_EXAMPLES * 2
    assert df["probability"].between(0.0, 1.0).all()


def test_multiclass_csv_has_three_classes(multiclass_encoded_data, multiclass_model, tmp_path):
    result_path = tmp_path / "multiclass"
    PathBuilder.build(result_path)
    label = Label("label1", [0, 1, 2])

    report = _make_report(result_path, multiclass_model, multiclass_encoded_data, multiclass_encoded_data, label)
    report._generate()

    df = pd.read_csv(result_path / "prob_report_test.csv")
    assert set(df["probability_for_class"].astype(str).unique()) == {"0", "1", "2"}
    assert len(df) == N_EXAMPLES * 3


def test_only_test_dataset_encoded(binary_encoded_data, binary_model, tmp_path):
    result_path = tmp_path / "test_only"
    PathBuilder.build(result_path)
    label = Label("label1", [0, 1], positive_class=1)

    train_ds = SequenceDataset()  # no encoded_data
    test_ds = SequenceDataset(encoded_data=binary_encoded_data)

    report = ClassProbabilityDistribution.build_object(
        name="prob_report",
        method=binary_model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        result_path=result_path,
        label=label
    )

    assert report.check_prerequisites()
    result = report._generate()

    assert len(result.output_figures) == 1
    assert result.output_figures[0].name == "Class probability distribution (test)"


def test_stratify_by_adds_column_and_generates_outputs(binary_encoded_data, binary_model, tmp_path):
    result_path = tmp_path / "stratify"
    PathBuilder.build(result_path)
    label = Label("label1", [0, 1], positive_class=1)

    dataset_with_meta = RandomDatasetGenerator.generate_sequence_dataset(
        N_EXAMPLES, {10: 1.0}, {"label1": {0: 0.5, 1: 0.5}, "cohort": {"A": 0.5, "B": 0.5}},
        result_path / "dataset"
    )
    dataset_with_meta.encoded_data = binary_encoded_data

    report = ClassProbabilityDistribution.build_object(
        name="prob_report",
        method=binary_model,
        train_dataset=dataset_with_meta,
        test_dataset=dataset_with_meta,
        result_path=result_path,
        label=label,
        stratify_by="cohort"
    )

    assert report.check_prerequisites()
    result = report._generate()

    assert isinstance(result, ReportResult)
    assert len(result.output_figures) == 2

    df = pd.read_csv(result_path / "prob_report_train.csv")
    assert "cohort" in df.columns
    assert set(df["cohort"].unique()).issubset({"A", "B"})


def test_check_prerequisites_fails_without_result_path(binary_encoded_data, binary_model):
    label = Label("label1", [0, 1], positive_class=1)
    report = ClassProbabilityDistribution.build_object(
        name="prob_report",
        method=binary_model,
        train_dataset=SequenceDataset(encoded_data=binary_encoded_data),
        test_dataset=SequenceDataset(encoded_data=binary_encoded_data),
        label=label
    )
    assert not report.check_prerequisites()


def test_check_prerequisites_fails_for_invalid_stratify_by(binary_encoded_data, binary_model, tmp_path):
    result_path = tmp_path / "bad_strat"
    PathBuilder.build(result_path)
    label = Label("label1", [0, 1], positive_class=1)

    report = _make_report(result_path, binary_model, binary_encoded_data, binary_encoded_data, label,
                          stratify_by="nonexistent_label")
    assert not report.check_prerequisites()
