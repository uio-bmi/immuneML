import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class MLSettingsPerformance(TrainMLModelReport):
    """
    Report for TrainMLModel instruction: plots the performance for each of the setting combinations as defined under 'settings' in the
    assessment (outer validation) loop.
    The performances are grouped by label (horizontal panels) encoding (vertical panels) and ML method (bar color).
    When multiple data splits are used, the average performance over the data splits is shown with an error bar
    representing the standard deviation.

    This report can be used only with TrainMLModel instruction under 'reports'.


    Arguments:

        single_axis_labels (bool): whether to use single axis labels. Note that using single axis labels makes the
        figure unsuited for rescaling, as the label position is given in a fixed distance from the axis. By default,
        single_axis_labels is False, resulting in standard plotly axis labels.

        x_label_position (float): if single_axis_labels is True, this should be an integer specifying the x axis label
        position relative to the x axis. The default value for label_position is -0.1.

        y_label_position (float): same as x_label_position, but for the y axis.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_hp_report: MLSettingsPerformance

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = "MLSettingsPerformance"

        single_axis_labels = kwargs["single_axis_labels"]
        ParameterValidator.assert_type_and_value(single_axis_labels, bool, location, "single_axis_labels")

        if single_axis_labels:
            x_label_position = kwargs["x_label_position"]
            ParameterValidator.assert_type_and_value(x_label_position, float, location, "x_label_position")
            y_label_position = kwargs["y_label_position"]
            ParameterValidator.assert_type_and_value(y_label_position, float, location, "y_label_position")
        else:
            x_label_position = None
            y_label_position = None

        name = kwargs["name"] if "name" in kwargs else None
        return MLSettingsPerformance(single_axis_labels, x_label_position, y_label_position, name)

    def __init__(self, single_axis_labels, x_label_position, y_label_position, name: str = None, state: TrainMLModelState = None, result_path: Path = None):
        super().__init__(name)

        self.single_axis_labels = single_axis_labels
        self.x_label_position = x_label_position
        self.y_label_position = y_label_position
        self.state = state
        self.result_path = None
        self.name = name
        self.result_name = "performance"
        self.vertical_grouping = "encoding"

    def _generate(self) -> ReportResult:
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
        filepath = self.result_path / f"{self.result_name}.csv"
        plotting_data.to_csv(filepath, index=False)
        return ReportOutput(filepath)

    def std(self, x):
        return x.std(ddof=0)

    def _plot(self, plotting_data):
        plotting_data = self._preprocess_plotting_data(plotting_data)

        metric_name = self.state.optimization_metric.name.replace("_", " ").title()

        if self.single_axis_labels:
            figure = self._plot_single_axis_labels(plotting_data, "ML method", f"Performance ({metric_name})")
        else:
            figure = self._plot_rescalable(plotting_data, "ML method", f"Performance<br>({metric_name})")

        file_path = self.result_path / f"{self.result_name}.html"
        figure.write_html(str(file_path))

        return ReportOutput(path=file_path)

    def _preprocess_plotting_data(self, plotting_data):
        plotting_data = plotting_data.groupby(["label", self.vertical_grouping, "ml_method"], as_index=False).agg(
            {"fold": "first", "performance": ['mean', self.std]})

        plotting_data.columns = plotting_data.columns.map(''.join)

        return plotting_data

    def _plot_rescalable(self, plotting_data, x_label, y_label):
        figure = px.bar(plotting_data, x="ml_method", y="performancemean", color="ml_method", barmode="relative",
                        facet_row=self.vertical_grouping, facet_col="label", error_y="performancestd",
                        labels={
                            "performancemean": y_label,
                            "ml_method": x_label,
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)
        figure.update_layout(showlegend=False)
        return figure

    def _plot_single_axis_labels(self, plotting_data, x_label, y_label):
        figure = self._plot_rescalable(plotting_data, x_label, y_label)
        return PlotlyUtil.add_single_axis_labels(figure, x_label, y_label, self.x_label_position, self.y_label_position)

    def check_prerequisites(self):
        run_report = True

        if self.state is None:
            warnings.warn(f"{self.__class__.__name__} can only be executed as a hyperparameter report. MLSettingsPerformance report will not be created.")
            run_report = False

        if self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output 'path' to be set. MLSettingsPerformance report will not be created.")
            run_report = False

        return run_report
