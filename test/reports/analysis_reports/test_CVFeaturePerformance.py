import os
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
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from source.reports.ReportResult import ReportResult
from source.reports.hyperparameter.CVFeaturePerformance import CVFeaturePerformance


class TestCVFeaturePerformance(TestCase):
    def test_generate(self):

        path = EnvironmentSettings.tmp_test_path + "cv_feature_performance/"

        state = HPOptimizationState(assessment=SplitConfig(split_count=5, split_strategy=SplitType.K_FOLD),
                                    selection=SplitConfig(split_count=10, split_strategy=SplitType.K_FOLD),
                                    optimization_metric=Metric.ACCURACY,
                                    label_configuration=LabelConfiguration(labels=[Label(name="CMV", values=[True, False])]),
                                    hp_settings=[HPSetting(encoder_params={"p_value_threshold": 0.001},
                                                           encoder=SequenceAbundanceEncoder, preproc_sequence=[],
                                                           ml_method=ProbabilisticBinaryClassifier(10, 0.1), ml_params={}),
                                                 HPSetting(encoder_params={"p_value_threshold": 0.01},
                                                           encoder=SequenceAbundanceEncoder, preproc_sequence=[],
                                                           ml_method=ProbabilisticBinaryClassifier(10, 0.1), ml_params={}),
                                                 HPSetting(encoder_params={"p_value_threshold": 0.01},
                                                           encoder=SequenceAbundanceEncoder, preproc_sequence=[],
                                                           ml_method=ProbabilisticBinaryClassifier(10, 0.01), ml_params={})
                                                 ], dataset=None, hp_strategy=None, metrics=None)

        report = CVFeaturePerformance("p_value_threshold", state, path)
        with self.assertWarns(RuntimeWarning):
            report.generate_report()

        state.hp_settings = state.hp_settings[:2]
        report_result = report.generate_report()

        self.assertTrue(isinstance(report_result, ReportResult))
        self.assertEqual(2, len(report_result.output_tables))
        self.assertEqual(1, len(report_result.output_figures))
        self.assertTrue(os.path.isfile(report_result.output_figures[0].path))
        self.assertTrue(os.path.isfile(report_result.output_tables[0].path))
        self.assertTrue(os.path.isfile(report_result.output_tables[1].path))

        shutil.rmtree(path)
