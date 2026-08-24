import os
import shutil

import pandas as pd
import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.reports.ml_reports.ClassProbabilityDistribution import ClassProbabilityDistribution
from immuneML.util.PathBuilder import PathBuilder


@pytest.fixture(autouse=True)
def set_test_cache():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name


def _write_predictions(path, label, true_classes, pos_proba):
    df = pd.DataFrame({
        "example_id": [f"ex_{i}" for i in range(len(true_classes))],
        f"{label.name}_true_class": true_classes,
        f"{label.name}_{label.positive_class}_proba": pos_proba,
        f"{label.name}_{label.get_binary_negative_class()}_proba": [1 - p for p in pos_proba],
    })
    filepath = path / "predictions.csv"
    df.to_csv(filepath, index=False)
    return filepath


def _make_report(result_path, label, predictions_path):
    return ClassProbabilityDistribution.build_object(
        name="prob_report", train_dataset=SequenceDataset(), test_dataset=SequenceDataset(),
        result_path=result_path, label=label, test_predictions_path=predictions_path)


def test_generates_figure_only():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "class_prob_dist_generate/")

    label = Label("label1", ["neg", "pos"], positive_class="pos")
    true_classes = ["pos" if i % 2 == 0 else "neg" for i in range(10)]
    pos_proba = [0.9 if cls == "pos" else 0.1 for cls in true_classes]
    predictions_path = _write_predictions(path, label, true_classes, pos_proba)

    report = _make_report(path, label, predictions_path)
    assert report.check_prerequisites()

    result = report._generate()

    assert len(result.output_figures) == 1
    assert result.output_tables == []
    assert (path / "prob_report_test.html").is_file()

    shutil.rmtree(path)


def test_binary_df_keeps_only_positive_class_probability():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "class_prob_dist_binary_df/")

    label = Label("label1", ["neg", "pos"], positive_class="pos")
    true_classes = ["pos" if i % 2 == 0 else "neg" for i in range(10)]
    pos_proba = [0.9 if cls == "pos" else 0.1 for cls in true_classes]
    predictions_path = _write_predictions(path, label, true_classes, pos_proba)

    report = _make_report(path, label, predictions_path)
    df, n_examples = report._build_proba_df(report.test_dataset, predictions_path)

    assert set(df.columns) == {"true_class", "probability"}
    assert len(df) == n_examples == len(true_classes)
    # a good classifier: positive-class probability is high for "pos" examples, low for "neg" examples
    assert (df[df["true_class"] == "pos"]["probability"] > 0.5).all()
    assert (df[df["true_class"] == "neg"]["probability"] < 0.5).all()

    shutil.rmtree(path)


def test_positive_class_is_not_swapped():
    """Regression test: the plotted probability must always come from the column matching
    label.positive_class, not whichever class happens to be encountered first in the data."""
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "class_prob_dist_positive/")

    label = Label("label1", ["neg", "pos"], positive_class="pos")
    true_classes = ["pos", "neg"]
    pos_proba = [0.8, 0.3]  # label1_pos_proba per example; label1_neg_proba is 1 - this
    predictions_path = _write_predictions(path, label, true_classes, pos_proba)

    report = _make_report(path, label, predictions_path)
    df, _ = report._build_proba_df(report.test_dataset, predictions_path)

    assert df[df["true_class"] == "pos"]["probability"].iloc[0] == pytest.approx(0.8)
    assert df[df["true_class"] == "neg"]["probability"].iloc[0] == pytest.approx(0.3)

    shutil.rmtree(path)


def test_check_prerequisites_fails_without_predictions():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "class_prob_dist_no_predictions/")

    label = Label("label1", ["neg", "pos"], positive_class="pos")
    report = ClassProbabilityDistribution.build_object(
        name="prob_report", train_dataset=SequenceDataset(), test_dataset=SequenceDataset(),
        result_path=path, label=label)
    assert not report.check_prerequisites()

    shutil.rmtree(path)