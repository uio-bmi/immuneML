import os
import shutil
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.HPLabelState import HPLabelState
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TopFeaturesPerformance import TopFeaturesPerformance
from immuneML.util.PathBuilder import PathBuilder


class TestTopFeaturesPerformance(TestCase):

    def setUp(self):
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _make_encoded_data(self, n_samples: int, n_features: int,
                           rng: np.random.Generator) -> EncodedData:
        X = rng.random((n_samples, n_features)).astype(np.float32)
        y = [bool(v) for v in rng.integers(0, 2, size=n_samples)]
        feature_names = [f"feat_{i}" for i in range(n_features)]
        return EncodedData(examples=X, labels={"cmv": y},
                           example_ids=[f"s{i}" for i in range(n_samples)],
                           feature_names=feature_names)

    def _make_method_mock(self, n_features: int, n_samples: int,
                          rng: np.random.Generator) -> MagicMock:
        method = MagicMock()
        # give the first 5 features clearly higher coefficients so top-N selection is stable
        coeffs = np.zeros(n_features)
        coeffs[:5] = rng.uniform(1.0, 2.0, size=5)
        coeffs[5:] = rng.uniform(0.0, 0.1, size=n_features - 5)
        method.get_params.return_value = {"coefficients": coeffs.tolist()}
        method.class_mapping = {0: False, 1: True}
        method.can_predict_proba.return_value = False
        method.model.predict.return_value = rng.integers(0, 2, size=n_samples)
        method.model.classes_ = [0, 1]
        return method

    def _make_hp_item(self, n_features: int, n_samples: int,
                      rng: np.random.Generator) -> HPItem:
        encoded_data = self._make_encoded_data(n_samples, n_features, rng)
        method = self._make_method_mock(n_features, n_samples, rng)
        dataset = MagicMock()
        dataset.encoded_data = encoded_data
        return HPItem(method=method, test_dataset=dataset)

    def _make_state(self, n_splits: int, n_features: int,
                    n_samples: int) -> TrainMLModelState:
        rng = np.random.default_rng(42)
        label = Label(name="cmv", values=[True, False], positive_class=True)
        label_config = LabelConfiguration(labels=[label])

        hp_setting = MagicMock()
        hp_setting.__str__ = MagicMock(return_value="mock_setting")

        assessment_states = []
        for i in range(n_splits):
            hp_item = self._make_hp_item(n_features, n_samples, rng)
            hp_item.hp_setting = hp_setting

            label_state = HPLabelState("cmv", [])
            label_state.assessment_items = {"mock_setting": hp_item}
            selection_state = MagicMock()
            selection_state.optimal_hp_setting = hp_setting
            label_state.selection_state = selection_state

            assessment_state = HPAssessmentState(i, None, None, None, label_config)
            assessment_state.label_states["cmv"] = label_state
            assessment_states.append(assessment_state)

        state = TrainMLModelState(
            assessment=SplitConfig(split_count=n_splits, split_strategy=SplitType.K_FOLD),
            selection=SplitConfig(split_count=1, split_strategy=SplitType.K_FOLD),
            optimization_metric=ClassificationMetric.BALANCED_ACCURACY,
            label_configuration=label_config,
            hp_settings=[], dataset=None, hp_strategy=None, metrics=None
        )
        state.assessment_states = assessment_states
        return state

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "top_features_performance/"
        PathBuilder.remove_old_and_build(path)

        n_splits = 3
        n_features = 10
        n_samples = 20
        top_n_values = [2, 5]

        state = self._make_state(n_splits, n_features, n_samples)
        report = TopFeaturesPerformance(top_n_values=top_n_values, use_optimal_only=True,
                                        metric=None, name="test_report", n_points=10,
                                        state=state, label=None, result_path=path)

        self.assertTrue(report.check_prerequisites())
        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        # two tables (performance CSV + frequency CSV) and two figures
        self.assertEqual(2, len(result.output_tables))
        self.assertEqual(3, len(result.output_figures))
        for out in result.output_tables + result.output_figures:
            self.assertTrue(out.path.is_file(), f"Missing output: {out.path}")

        # --- performance CSV ---
        perf_csv = next(o for o in result.output_tables if "performance" in o.path.name)
        df_perf = pd.read_csv(perf_csv.path)
        for col in ("n_features", "performance", "split", "is_full_model", "in_explicit_set"):
            self.assertIn(col, df_perf.columns)

        # CSV must contain the union of explicit values, log-spaced values, and the full model
        _log_pts = np.logspace(0, np.log10(n_features), 10)
        _log_ns = {int(round(x)) for x in _log_pts if 1 <= round(x) <= n_features}
        expected_ns = sorted(set(top_n_values) | _log_ns | {n_features})
        self.assertEqual(set(expected_ns), set(df_perf["n_features"].unique()))
        # one row per split × n_value
        self.assertEqual(n_splits * len(expected_ns), len(df_perf))
        # performance values must be valid metric scores [0, 1]
        self.assertTrue(df_perf["performance"].between(0.0, 1.0).all())
        # exactly one entry per split is marked as full model
        full_model_rows = df_perf[df_perf["is_full_model"]]
        self.assertEqual(n_splits, len(full_model_rows))
        self.assertTrue((full_model_rows["n_features"] == n_features).all())
        # in_explicit_set must be True exactly for top_n_values, False for everything else
        self.assertTrue(df_perf[df_perf["n_features"].isin(top_n_values)]["in_explicit_set"].all())
        self.assertFalse(df_perf[~df_perf["n_features"].isin(top_n_values)]["in_explicit_set"].any())

        # --- frequency CSV ---
        freq_csv = next(o for o in result.output_tables if "frequency" in o.path.name)
        df_freq = pd.read_csv(freq_csv.path)
        self.assertIn("feature", df_freq.columns)
        self.assertIn("selection_count", df_freq.columns)
        self.assertIn("n_features", df_freq.columns)
        self.assertIn("n_splits", df_freq.columns)

        # n_splits column must always equal the actual split count
        self.assertTrue((df_freq["n_splits"] == n_splits).all())
        # selection counts must be in [1, n_splits]
        self.assertTrue(df_freq["selection_count"].between(1, n_splits).all())
        # only top_n_values (not full model) appear in frequency data
        self.assertEqual({2, 5}, set(df_freq["n_features"].unique()))
        # features must come from the known feature set
        known_features = {f"feat_{i}" for i in range(n_features)}
        self.assertTrue(set(df_freq["feature"].unique()).issubset(known_features))

        # shutil.rmtree(path)