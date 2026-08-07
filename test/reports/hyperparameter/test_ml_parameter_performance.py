import os
import shutil
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.HPLabelState import HPLabelState
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.MLParameterPerformance import MLParameterPerformance
from immuneML.util.PathBuilder import PathBuilder


class TestMLParameterPerformance(TestCase):

    def setUp(self):
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _make_label_config(self) -> LabelConfiguration:
        return LabelConfiguration(labels=[Label(name="cmv", values=[True, False], positive_class=True)])

    def _make_method_mock(self, n_nonzero: int, n_features: int) -> MagicMock:
        method = MagicMock()
        coefficients = [1.0] * n_nonzero + [0.0] * (n_features - n_nonzero)
        method.get_params.return_value = {"coefficients": coefficients}
        return method

    def _make_state(self, hp_values: list, n_splits: int, n_features: int,
                    hp_parameter_name: str = "C", metrics: tuple = ("balanced_accuracy",)) -> TrainMLModelState:
        label_config = self._make_label_config()
        rng = np.random.default_rng(0)

        hp_settings = [HPSetting(encoder=None, encoder_params={}, ml_method=None, ml_params={hp_parameter_name: v},
                                 preproc_sequence=[], ml_method_name=f"ml_{i}", encoder_name="enc")
                      for i, v in enumerate(hp_values)]

        assessment_states = []
        for split_idx in range(n_splits):
            label_state = HPLabelState("cmv", [])
            for i, hp_setting in enumerate(hp_settings):
                # more non-zero coefficients for later settings, so the path is monotonic-ish
                n_nonzero = min(n_features, i * 2 + 1)
                method = self._make_method_mock(n_nonzero, n_features)
                performance = {m: float(rng.uniform(0.5, 1.0)) for m in metrics}
                label_state.assessment_items[hp_setting.get_key()] = HPItem(
                    method=method, hp_setting=hp_setting, performance=performance)

            assessment_state = HPAssessmentState(split_idx, None, None, None, label_config)
            assessment_state.label_states["cmv"] = label_state
            assessment_states.append(assessment_state)

        state = TrainMLModelState(
            assessment=SplitConfig(split_count=n_splits, split_strategy=SplitType.K_FOLD),
            selection=SplitConfig(split_count=1, split_strategy=SplitType.K_FOLD),
            optimization_metric=ClassificationMetric.BALANCED_ACCURACY,
            label_configuration=label_config, hp_settings=hp_settings, dataset=None, hp_strategy=None, metrics=None)
        state.assessment_states = assessment_states
        return state

    def test_generate_numeric_with_multiple_metrics(self):
        path = EnvironmentSettings.tmp_test_path / "ml_parameter_performance_numeric/"
        PathBuilder.remove_old_and_build(path)

        c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
        n_splits, n_features = 3, 10
        state = self._make_state(c_values, n_splits, n_features, metrics=("balanced_accuracy", "auc"))

        report = MLParameterPerformance(hp_parameter_name="C", metrics=["BALANCED_ACCURACY", "AUC"],
                                        name="test_report", state=state, result_path=path)

        self.assertTrue(report.check_prerequisites())
        result = report._generate()

        self.assertIsInstance(result, ReportResult)
        # one CSV per metric
        self.assertEqual(2, len(result.output_tables))
        # two figures per metric (vs. hp_value, vs. n_features), since coefficients are available
        self.assertEqual(4, len(result.output_figures))
        for out in result.output_tables + result.output_figures:
            self.assertTrue(out.path.is_file(), f"Missing output: {out.path}")

        df = pd.read_csv(result.output_tables[0].path)
        for col in ("split", "hp_setting", "hp_value", "n_features", "performance", "label"):
            self.assertIn(col, df.columns)

        # one row per split x hp_setting
        self.assertEqual(n_splits * len(c_values), len(df))
        self.assertEqual(set(c_values), set(df["hp_value"].unique()))
        # n_features grows with the C value by construction
        by_c = df.groupby("hp_value")["n_features"].mean().sort_index()
        self.assertTrue(all(np.diff(by_c.values) >= 0))

        shutil.rmtree(path)

    def test_generate_categorical_hp_value(self):
        path = EnvironmentSettings.tmp_test_path / "ml_parameter_performance_categorical/"
        PathBuilder.remove_old_and_build(path)

        state = self._make_state(["adam", "sgd"], n_splits=3, n_features=5, hp_parameter_name="optimizer")

        report = MLParameterPerformance(hp_parameter_name="optimizer", metrics=None,
                                        name="test_report", state=state, result_path=path)

        self.assertTrue(report.check_prerequisites())
        result = report._generate()

        self.assertEqual(1, len(result.output_tables))
        # both the hp_value plot and the n_features plot should still be produced
        self.assertEqual(2, len(result.output_figures))
        for out in result.output_tables + result.output_figures:
            self.assertTrue(out.path.is_file())

        df = pd.read_csv(result.output_tables[0].path)
        self.assertEqual({"adam", "sgd"}, set(df["hp_value"].unique()))

        shutil.rmtree(path)

    def test_hp_settings_without_parameter_are_excluded(self):
        path = EnvironmentSettings.tmp_test_path / "ml_parameter_performance_filter/"
        PathBuilder.remove_old_and_build(path)

        label_config = self._make_label_config()
        with_param = HPSetting(encoder=None, encoder_params={}, ml_method=None, ml_params={"C": 1.0},
                               preproc_sequence=[], ml_method_name="ml_with", encoder_name="enc")
        without_param = HPSetting(encoder=None, encoder_params={}, ml_method=None, ml_params={"alpha": 0.5},
                                  preproc_sequence=[], ml_method_name="ml_without", encoder_name="enc")

        label_state = HPLabelState("cmv", [])
        label_state.assessment_items = {
            with_param.get_key(): HPItem(method=self._make_method_mock(1, 5), hp_setting=with_param,
                                         performance={"balanced_accuracy": 0.7}),
            without_param.get_key(): HPItem(method=self._make_method_mock(2, 5), hp_setting=without_param,
                                            performance={"balanced_accuracy": 0.8}),
        }
        assessment_state = HPAssessmentState(0, None, None, None, label_config)
        assessment_state.label_states["cmv"] = label_state

        state = TrainMLModelState(
            assessment=SplitConfig(split_count=1, split_strategy=SplitType.K_FOLD),
            selection=SplitConfig(split_count=1, split_strategy=SplitType.K_FOLD),
            optimization_metric=ClassificationMetric.BALANCED_ACCURACY, label_configuration=label_config,
            hp_settings=[with_param, without_param], dataset=None, hp_strategy=None, metrics=None)
        state.assessment_states = [assessment_state]

        report = MLParameterPerformance(hp_parameter_name="C", metrics=None, name="test_report",
                                        state=state, result_path=path)
        result = report._generate()

        df = pd.read_csv(result.output_tables[0].path)
        self.assertEqual(1, len(df))
        self.assertEqual(with_param.get_key(), df.iloc[0]["hp_setting"])

        shutil.rmtree(path)

    def test_missing_metric_is_skipped_without_crashing(self):
        path = EnvironmentSettings.tmp_test_path / "ml_parameter_performance_missing_metric/"
        PathBuilder.remove_old_and_build(path)

        state = self._make_state([0.1, 1.0], n_splits=2, n_features=5, metrics=("balanced_accuracy",))

        report = MLParameterPerformance(hp_parameter_name="C", metrics=["AUC"], name="test_report",
                                        state=state, result_path=path)
        result = report._generate()

        # requested metric (AUC) was never computed for any HP setting, so nothing is produced, but no crash
        self.assertEqual(0, len(result.output_tables))
        self.assertEqual(0, len(result.output_figures))

        shutil.rmtree(path)