import warnings

import pandas as pd
import plotly.express as px

from source.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder


class MLSettingsPerformance(MLReport):
    """
    Report for TrainMLModel instruction: plots the performance for each of the setting combinations as defined under 'settings' in the
    assessment (outer validation) loop.
    The performances are grouped by label (horizontal panels) encoding (vertical panels) and ML method (bar color).
    When multiple data splits are used, the average performance over the data splits is shown with an error bar
    representing the standard deviation.

    This report can be used only with TrainMLModel instruction under assessment/reports/hyperparameter.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_hp_report: MLSettingsPerformance

    """

    @classmethod
    def build_object(cls, **kwargs):
        return MLSettingsPerformance(kwargs["name"] if "name" in kwargs else None)

    def __init__(self, name: str = None, state: TrainMLModelState = None, result_path: str = None):
        super(MLSettingsPerformance, self).__init__()

        self.state = state
        self.result_path = None
        self.name = name
        self.result_name = "performance"
        self.vertical_grouping = "encoding"

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        plotting_data = self._retrieve_plotting_data()
        result_table = self._write_results_table(plotting_data)
        report_output_fig = self._safe_plot(plotting_data=plotting_data)
        output_figures = [report_output_fig] if report_output_fig is not None else []

        return ReportResult(self.name, output_tables=[result_table], output_figures=output_figures)

    def _get_vertical_grouping(self, assessment_item):
        return assessment_item.hp_setting.encoder_name

    def _get_color_grouping(self, assessment_item):
        return assessment_item.hp_setting.ml_method_name

    def _retrieve_plotting_data(self):
        plotting_data = []

        for assessment_state in self.state.assessment_states:
            for label_key, label_state in assessment_state.label_states.items():
                for assessment_key, assessment_item in label_state.assessment_items.items():
                    plotting_data.append([assessment_state.split_index,
                                          label_key,
                                          self._get_vertical_grouping(assessment_item),
                                          self._get_color_grouping(assessment_item),
                                          assessment_item.performance[self.state.optimization_metric.name.lower()]])
                    # optional: include assessment_item.hp_setting.preproc_sequence_name. for now ignored.

        plotting_data = pd.DataFrame(plotting_data, columns=["fold", "label", self.vertical_grouping, "ml_method", "performance"])
        plotting_data.replace(to_replace=[None], value="NA", inplace=True)

        return plotting_data

    def _write_results_table(self, plotting_data):
        filepath = f"{self.result_path}{self.result_name}.csv"
        plotting_data.to_csv(filepath, index=False)
        return ReportOutput(filepath)

    def std(self, x):
        return x.std(ddof=0)

    def _plot(self, plotting_data):
        plotting_data = plotting_data.groupby(["label",  self.vertical_grouping, "ml_method"], as_index=False).agg(
            {"fold": "first", "performance": ['mean', self.std]})

        plotting_data.columns = plotting_data.columns.map(''.join)

        metric_name = self.state.optimization_metric.name.replace("_", " ").title()

        figure = px.bar(plotting_data, x="ml_method", y="performancemean", color="ml_method", barmode="relative",
                        facet_row=self.vertical_grouping, facet_col="label", error_y="performancestd",
                        labels={
                            "performancemean": f"Performance<br>({metric_name})",
                            "ml_method": "ML method"
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = f"{self.result_path}{self.result_name}.html"
        figure.write_html(file_path)

        return ReportOutput(path=file_path)

    def check_prerequisites(self):
        run_report = True

        if self.state is None:
            warnings.warn(f"{self.__class__.__name__} can only be executed as a hyperparameter report. MLSettingsPerformance report will not be created.")
            run_report = False

        if self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output 'path' to be set. MLSettingsPerformance report will not be created.")
            run_report = False

        return run_report
