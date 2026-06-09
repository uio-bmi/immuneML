import os

import numpy as np
import pandas as pd
import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.ml_methods.classifiers.RandomForestClassifier import RandomForestClassifier
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.SHAPReport import SHAPReport
from immuneML.util.PathBuilder import PathBuilder

N_FEATURES = 10
FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]
LABEL = Label("disease", [0, 1])


@pytest.fixture(autouse=True)
def set_cache_type():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name


def _encoded_data(n_samples, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, N_FEATURES))
    y = (rng.random(n_samples) > 0.5).astype(int).tolist()
    return EncodedData(examples=X, labels={"disease": y}, feature_names=FEATURE_NAMES)


def _make_report(method, train_enc, test_enc, result_path, **kwargs):
    report = SHAPReport.build_object(**{"n_background_samples": 20, "plot_top_n_features": 5, **kwargs})
    report.method = method
    report.label = LABEL
    report.result_path = result_path
    report.train_dataset = Dataset()
    report.train_dataset.encoded_data = train_enc
    report.test_dataset = Dataset()
    report.test_dataset.encoded_data = test_enc
    return report


@pytest.fixture
def trained_lr():
    model = LogisticRegression()
    model.fit(_encoded_data(50), LABEL)
    return model


@pytest.fixture
def trained_rf():
    model = RandomForestClassifier()
    model.fit(_encoded_data(50), LABEL)
    return model


def test_generate_logistic_regression(tmp_path, trained_lr):
    path = PathBuilder.build(tmp_path / "shap_lr")
    report = _make_report(trained_lr, _encoded_data(50), _encoded_data(20, seed=7), path)

    assert report.check_prerequisites()
    result = report._generate()

    assert isinstance(result, ReportResult)
    assert result.output_tables[0].path == path / "shap_values.csv"
    assert result.output_figures[0].path == path / "shap_beeswarm.html"
    assert result.output_figures[1].path == path / "shap_mean_bar.html"
    assert (path / "shap_values.csv").is_file()
    assert (path / "shap_beeswarm.html").is_file()
    assert (path / "shap_mean_bar.html").is_file()

    df = pd.read_csv(path / "shap_values.csv")
    assert list(df.columns) == FEATURE_NAMES
    assert len(df) == 20


def test_generate_random_forest(tmp_path, trained_rf):
    path = PathBuilder.build(tmp_path / "shap_rf")
    report = _make_report(trained_rf, _encoded_data(50), _encoded_data(20, seed=7), path)

    assert report.check_prerequisites()
    result = report._generate()

    assert isinstance(result, ReportResult)
    assert (path / "shap_values.csv").is_file()


def test_generate_falls_back_to_train_without_test_dataset(tmp_path, trained_lr):
    path = PathBuilder.build(tmp_path / "shap_notrain")
    report = SHAPReport.build_object(n_background_samples=20, plot_top_n_features=5)
    report.method = trained_lr
    report.label = LABEL
    report.result_path = path
    report.train_dataset = Dataset()
    report.train_dataset.encoded_data = _encoded_data(50)
    report.test_dataset = None

    result = report._generate()
    assert isinstance(result, ReportResult)
    df = pd.read_csv(path / "shap_values.csv")
    assert len(df) == 50


def test_plot_top_n_none_shows_all_features(tmp_path, trained_lr):
    path = PathBuilder.build(tmp_path / "shap_all")
    report = _make_report(trained_lr, _encoded_data(50), _encoded_data(20, seed=7), path,
                          plot_top_n_features=None)
    report._generate()

    df = pd.read_csv(path / "shap_values.csv")
    assert len(df.columns) == N_FEATURES


def test_check_prerequisites_fails_for_non_sklearn_method(tmp_path):
    from unittest.mock import MagicMock
    report = SHAPReport()
    report.method = MagicMock(spec=[])
    report.train_dataset = Dataset()
    report.train_dataset.encoded_data = MagicMock()
    assert not report.check_prerequisites()


def test_check_prerequisites_fails_without_encoded_data(tmp_path, trained_lr):
    report = SHAPReport()
    report.method = trained_lr
    report.train_dataset = Dataset()
    report.train_dataset.encoded_data = None
    assert not report.check_prerequisites()