import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.PathBuilder import PathBuilder


class CVFeaturePerformance(TrainMLModelReport):
    """
    This report plots the average training vs test performance w.r.t. given encoding parameter which is explicitly set
    in the feature attribute. It can be used only in combination with TrainMLModel instruction and can be only specified under 'reports'


    Arguments:
        feature: name of the encoder parameter w.r.t. which the performance across training and test will be shown. Possible values depend
        on the encoder on which it is used.

        is_feature_axis_categorical (bool): if the x-axis of the plot where features are shown should be categorical; alternatively it is
        automatically determined based on the feature values


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        report1:
            CVFeaturePerformance:
                feature: p_value_threshold # parameter value of SequenceAbundance encoder
                is_feature_axis_categorical: True # show x-axis as categorical

    """

    @classmethod
    def build_object(cls, **kwargs):
        return CVFeaturePerformance(**kwargs)

    def __init__(self, feature: str = None, state: TrainMLModelState = None, result_path: Path = None, label: str = None,
                 name: str = None, is_feature_axis_categorical: bool = None):
        super().__init__(name)
        self.feature = feature
        self.state = state
        self.result_path = result_path
        self.label = label
        self.relevant_hp_settings = []
        self.feature_values = None
        self.feature_count = None
        self.name = name
        self.result_name = None
        self.is_feature_axis_categorical = is_feature_axis_categorical

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
        if self.label is None and len(self.state.label_configuration.get_labels_by_name()) == 1:
            self.label = self.state.label_configuration.get_labels_by_name()[0]

    def _extract_hp_settings(self):
        self.relevant_hp_settings = [hp_setting for hp_setting in self.state.hp_settings
                                     if self.feature in hp_setting.encoder_params]
        self.feature_values = np.unique([hp_setting.encoder_params[self.feature] for hp_setting in self.relevant_hp_settings])
        self.feature_count = len(self.feature_values)

    def _generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)
        self.result_name = f"{self.feature}_performance"

        training_dataframe, test_dataframe = self._make_plot_dataframes()
        table_results = self._store_dataframes(training_dataframe, test_dataframe)

        report_output_fig = self._plot(training_dataframe=training_dataframe, test_dataframe=test_dataframe)
        output_figures = None if report_output_fig is None else [report_output_fig]

        return ReportResult(output_tables=table_results,
                            output_figures=output_figures)

    def _plot(self, training_dataframe, test_dataframe):

        optimization_metric = self.state.optimization_metric.name.lower()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=training_dataframe["x"], y=training_dataframe["y"], name="training", mode="markers", marker_size=11, marker_color="#CC79A7",
                                 hovertemplate=f"training {optimization_metric}" + ": %{y}<extra></extra>", opacity=0.8))
        fig.add_trace(go.Scatter(x=test_dataframe["x"], y=test_dataframe["y"], name="test", mode="markers", marker_size=11, marker_color="#009E73",
                                 hovertemplate=f"test {optimization_metric}" + ": %{y}<extra></extra>", opacity=0.8))
        fig.update_layout(legend_title_text="Data", title="Performance across feature values", template="plotly_white")
        fig.update_xaxes(title_text=self.feature)
        if self.is_feature_axis_categorical:
            fig.update_xaxes(type='category')
        fig.update_yaxes(title_text=f"performance ({self.state.optimization_metric.name.lower()})")
        fig.update_layout(hovermode="x unified")

        file_path = self.result_path / f"{self.result_name}.html"
        fig.write_html(str(file_path))

        return ReportOutput(path=file_path)

    def _store_dataframes(self, training_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame) -> List[ReportOutput]:
        train_path = self.result_path / "training_performance.csv"
        test_path = self.result_path / "test_performance.csv"
        training_dataframe.to_csv(train_path, index=False)
        test_dataframe.to_csv(test_path, index=False)

        return [ReportOutput(path=train_path, name=f"Training performance w.r.t. {self.feature} values"),
                ReportOutput(path=test_path, name=f"Test performance w.r.t. {self.feature} values")]

    def _make_plot_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        performance_training = np.zeros((self.feature_count, self.state.assessment.split_count,
                                         self.state.selection.split_count))
        features_test = np.zeros((self.state.assessment.split_count, self.feature_count))
        performance_test = np.zeros((self.state.assessment.split_count, self.feature_count))

        for assessment_split_index, assessment_state in enumerate(self.state.assessment_states):

            assessment_items = [assessment_state.label_states[self.label].assessment_items[hp_setting.get_key()]
                                for hp_setting in self.relevant_hp_settings]
            features_test[assessment_split_index] = [item.hp_setting.encoder_params[self.feature] for item in assessment_items]
            performance_test[assessment_split_index] = [item.performance[self.state.optimization_metric.name.lower()] for item in assessment_items]

            for hp_index, hp_setting in enumerate(self.relevant_hp_settings):
                performance_training[hp_index, assessment_split_index] = \
                    [item.performance[self.state.optimization_metric.name.lower()]
                     for item in assessment_state.label_states[self.label].selection_state.hp_items[hp_setting.get_key()]]

        feature_values = self.feature_values.astype(str)

        test_dataframe = pd.DataFrame({"x": feature_values, "y": performance_test.mean(axis=0)})
        training_dataframe = pd.DataFrame({"x": feature_values, "y": performance_training.mean(axis=(1, 2))})

        return training_dataframe, test_dataframe
