import os
import random
import shutil
from unittest import TestCase

from source.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from source.hyperparameter_optimization.states.HPItem import HPItem
from source.hyperparameter_optimization.states.HPLabelState import HPLabelState
from source.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from source.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from source.hyperparameter_optimization.strategy.GridSearch import GridSearch
from source.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from source.reports.ReportResult import ReportResult
from source.reports.hyperparameter_reports.CVFeaturePerformance import CVFeaturePerformance


class TestCVFeaturePerformance(TestCase):
    def test_generate(self):

        path = EnvironmentSettings.tmp_test_path + "cv_feature_performance/"

        state = TrainMLModelState(assessment=SplitConfig(split_count=5, split_strategy=SplitType.K_FOLD),
                                  selection=SplitConfig(split_count=10, split_strategy=SplitType.K_FOLD),
                                  optimization_metric=Metric.ACCURACY,
                                  label_configuration=LabelConfiguration(labels=[Label(name="CMV", values=[True, False])]),
                                  hp_settings=[HPSetting(encoder_params={"p_value_threshold": 0.001}, encoder_name="e1",
                                                           encoder=SequenceAbundanceEncoder, preproc_sequence=[], ml_method_name="ml1",
                                                           ml_method=ProbabilisticBinaryClassifier(10, 0.1), ml_params={}),
                                                 HPSetting(encoder_params={"p_value_threshold": 0.01}, encoder_name="e2",
                                                           encoder=SequenceAbundanceEncoder, preproc_sequence=[], ml_method_name="ml1",
                                                           ml_method=ProbabilisticBinaryClassifier(10, 0.1), ml_params={}),
                                                 HPSetting(encoder_params={"p_value_threshold": 0.01},
                                                           encoder=SequenceAbundanceEncoder, preproc_sequence=[],
                                                           ml_method=ProbabilisticBinaryClassifier(10, 0.01), ml_params={})
                                                 ], dataset=None, hp_strategy=None, metrics=None)

        report = CVFeaturePerformance("p_value_threshold", state, path, is_feature_axis_categorical=True, name="report1")
        with self.assertWarns(RuntimeWarning):
            report.generate_report()

        state.hp_settings = state.hp_settings[:2]

        state.assessment_states = [HPAssessmentState(i, None, None, None, state.label_configuration) for i in range(state.assessment.split_count)]
        for assessment_state in state.assessment_states:
            assessment_state.label_states["CMV"] = HPLabelState("CMV", [])
            assessment_state.label_states["CMV"].assessment_items = {setting.get_key(): HPItem(performance={'accuracy': random.uniform(0.5, 1)}, hp_setting=setting)
                                                                     for setting in state.hp_settings}
            assessment_state.label_states["CMV"].selection_state = HPSelectionState([], [], "", GridSearch(state.hp_settings))
            assessment_state.label_states["CMV"].selection_state.hp_items = {setting.get_key(): [HPItem(performance={'accuracy': random.uniform(0.5, 1)}, hp_setting=setting) for _ in range(state.selection.split_count)]
                                                                             for setting in state.hp_settings}

        report.state = state

        report.check_prerequisites()
        report_result = report.generate()

        self.assertTrue(isinstance(report_result, ReportResult))
        self.assertEqual(2, len(report_result.output_tables))
        self.assertEqual(1, len(report_result.output_figures))
        self.assertTrue(os.path.isfile(report_result.output_figures[0].path))
        self.assertTrue(os.path.isfile(report_result.output_tables[0].path))
        self.assertTrue(os.path.isfile(report_result.output_tables[1].path))

        shutil.rmtree(path)
