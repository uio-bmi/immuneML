import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.reports.Report import Report
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.util.PathBuilder import PathBuilder


class CVFeaturePerformance(Report):
    """
    Plots average training vs test performance w.r.t. given encoding parameter which is explicitly set
    in the feature attribute

    Attributes:
        feature: name of the encoder parameter w.r.t. which the performance across training and test
                 will be shown

    Specification:

        definitions:
            datasets:
                my_data:
                    ...
            encodings:
                enc1: SequenceAbundanceEncoder
                enc2:
                    SequenceAbundanceEncoder:
                        p_value_threshold: 0.01
            reports:
                report1:
                    CVFeaturePerformance:
                        feature: p_value_threshold
                        label: CMV
            ml_methods:
                ml1: ProbabilisticBinaryClassifier

        instructions:
            instruction_1:
                type: HPOptimization
                settings: [{encoding: enc1, ml_method: ml1]
                dataset: my_data
                assessment:
                    split_strategy: random
                    split_count: 1
                    training_percentage: 0.7
                    reports:
                        hyperparameter: [r1]
                selection:
                    split_strategy: random
                    split_count: 1
                    training_percentage: 0.7
                    reports:
                        model:
                            - report1
                labels:
                  - CMV
                strategy: GridSearch
                metrics: [accuracy, auc]
                optimization_metric: accuracy
                batch_size: 4
                reports: []
    """

    @classmethod
    def build_object(cls, **kwargs):
        return CVFeaturePerformance(**kwargs)

    def __init__(self, feature: str = None, hp_optimization_state: HPOptimizationState = None, result_path: str = None, label: str = None):
        super().__init__()
        self.feature = feature
        self.hp_optimization_state = hp_optimization_state
        self.result_path = result_path
        self.label = label
        self.relevant_hp_settings = []
        self.feature_values = None
        self.feature_count = None

    def check_prerequisites(self):
        self._extract_label()

        if self.label is None:
            warnings.warn("CVFeaturePerformance: the label was not set for this report and it could not be inferred from the instruction "
                          "as there might be multiple labels there. Skipping the report.", RuntimeWarning)
            return False

        self._extract_hp_settings()
        if self.feature_count != len(self.relevant_hp_settings):
            warnings.warn(f"CVFeaturePerformance: there are multiple hyperparameter settings with the same value of the "
                          f"feature {self.feature}. Skipping the report...", RuntimeWarning)
            return False

        return True

    def _extract_label(self):
        if self.label is None and len(self.hp_optimization_state.label_configuration.get_labels_by_name()) == 1:
            self.label = self.hp_optimization_state.label_configuration.get_labels_by_name()[0]

    def _extract_hp_settings(self):
        self.relevant_hp_settings = [hp_setting for hp_setting in self.hp_optimization_state.hp_settings
                                     if self.feature in hp_setting.encoder_params]
        self.feature_values = np.unique([hp_setting.encoder_params[self.feature] for hp_setting in self.relevant_hp_settings])
        self.feature_count = len(self.feature_values)

    def generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)
        result_name = f"{self.feature}_performance"

        training_dataframe, test_dataframe = self._make_plot_dataframes()
        table_results = self._store_dataframes(training_dataframe, test_dataframe)
        pandas2ri.activate()

        with open(EnvironmentSettings.visualization_path + "Scatterplot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_two_dataframes(df1=training_dataframe, df2=test_dataframe, label1="training", label2="test", x_label=self.feature,
                                 y_label=f"performance ({self.hp_optimization_state.optimization_metric.name.lower()})",
                                 result_path=self.result_path, result_name=result_name)

        return ReportResult(output_tables=table_results, output_figures=[ReportOutput(path=f"{self.result_path}{result_name}.pdf")])

    def _store_dataframes(self, training_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame) -> List[ReportOutput]:
        train_path = self.result_path + "training_performance.csv"
        test_path = self.result_path + "test_performance.csv"
        training_dataframe.to_csv(train_path, index=False)
        test_dataframe.to_csv(test_path, index=False)

        return [ReportOutput(path=train_path, name=f"Training performance w.r.t. {self.feature} values"),
                ReportOutput(path=test_path, name=f"Test performance w.r.t. {self.feature} values")]

    def _make_plot_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        performance_training = np.zeros((self.feature_count, self.hp_optimization_state.assessment.split_count,
                                         self.hp_optimization_state.selection.split_count))
        features_test = np.zeros((self.hp_optimization_state.assessment.split_count, self.feature_count))
        performance_test = np.zeros((self.hp_optimization_state.assessment.split_count, self.feature_count))

        for assessment_split_index, assessment_state in enumerate(self.hp_optimization_state.assessment_states):

            assessment_items = [assessment_state.label_states[self.label].assessment_items[hp_setting]
                                for hp_setting in self.relevant_hp_settings]
            features_test[assessment_split_index] = [item.hp_setting.encoder_params[self.feature] for item in assessment_items]
            performance_test[assessment_split_index] = [item.performance for item in assessment_items]

            for hp_index, hp_setting in enumerate(self.relevant_hp_settings):
                performance_training[hp_index, assessment_split_index] = \
                    [item.performance for item in assessment_state.label_states[self.label].selection_state.hp_items[str(hp_setting)]]

        feature_values = self.feature_values.astype(str)

        test_dataframe = pd.DataFrame({"x": feature_values, "y": performance_test.mean(axis=0)})
        training_dataframe = pd.DataFrame({"x": feature_values, "y": performance_training.mean(axis=(1, 2))})

        return training_dataframe, test_dataframe
