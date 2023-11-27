import plotly.express as px
import warnings
from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper

class WeightsDistribution(DataReport):
    """
    Plots the distribution of weights in a given Dataset. This report can only be used if example weighting has been applied to the given dataset.


    # todo: the report should work with any label, with any number of classes (currently assumes is_binder with classes 0 and 1)
    # use self.label property to find the label name.
    # do not hardcode any text related to the is_binder label

    # todo: make sure the same classes always get the same color in the figures
    # currently when running the report multiple times, class 0 is sometimes blue and sometimes purple
    # solution: retrieve the classes, sort the classes, use color_discrete_map instead of color_discrete_sequence
    # to map each class to a color. Get these colors for example from px.colors.diverging.Tealrose

    # todo: label is only a useful parameter when split_classes is true. these parameters can thus be merged
    # instead, use parameter color_grouping_label. if set, use different colors. if not set, use one color
    # see FeatureComparison report for an example where multiple labels can be used to change features of the plot

    # todo add unit tests
    # todo add the sequence as one of the hover values (plotly argument hover_data=["..."]), see example: https://plotly.com/python/hover-text-and-formatting/


    Example YAML specification:
        r1:
      WeightsDistribution:
        label:
          is_binding
        weight_thresholds:
        - 1
        - 0.1
        - 0.001
        split_classes:
          True
    """
    @classmethod
    def build_object(cls, **kwargs):
        return WeightsDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None, label: dict = None, weight_thresholds: dict = None, split_classes: bool = None):
            super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
            self.label = label
            self.weight_thresholds = weight_thresholds
            self.split_classes = split_classes
            self.label_config = None

    def check_prerequisites(self):
        if self.dataset.get_example_weights() is not None:
            return True
        else:
            warnings.warn("WeightsDistribution: report requires weights to be set for the given Dataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:
        self.label_config = LabelHelper.create_label_config([self.label], self.dataset, WeightsDistribution.__name__,
                                                       f"{WeightsDistribution.__name__}/label")
        data = self._get_plotting_data()
        report_output_fig = self._safe_plot(data=data)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(name=self.name,
                                info="A line graph of weights over sequences for is-binding and non-binding sequences",
                                output_figures=output_figures)


    def get_weights_by_class_df(self, data, weights):
        data["weights"] = weights
        data.sort_values(by=["weights"], inplace=True, ascending=False)

        data["sorted"] = None
        data.loc[data["is_binding"] == "0", "sorted"] = range(len(data[data["is_binding"] == "0"]))
        data.loc[data["is_binding"] == "1", "sorted"] = range(len(data[data["is_binding"] == "1"]))

        return data

    def get_weights_both_classes_df(self, data, weights):
        data["weights"] = weights
        data.sort_values(by=["weights"], inplace=True, ascending=False)
        data.loc[:, "sorted"] = range(len(data))

        return data

    def _get_plotting_data(self):
        weights = self.dataset.get_example_weights()
        data = self.dataset.get_metadata([self.label], return_df=True)

        # todo remove this, this was temporary
        if isinstance(self.dataset, SequenceDataset):
            data["seq"] = [seq.sequence_aa for seq in self.dataset.get_data()]

        if self.split_classes:
            data = self.get_weights_by_class_df(data, weights)
        else:
            data = self.get_weights_both_classes_df(data, weights)

        return data

    def _plot(self, data) -> ReportOutput:
        print(data)
        PathBuilder.build(self.result_path)

        if self.split_classes:
            color = self.label
        else:
            color = None

        fig = px.line(data, y="weights", color=color, x="sorted", log_y=True,
                      color_discrete_sequence=["#65D0B8", "#AB80D8"],
                      hover_data=["seq"],
                      labels={
                          "weights": "Weights",
                          "sorted": "Sequences (sorted by weight)",
                          "is_binding": "Is binder"
                      }, template="plotly_white")

        if self.weight_thresholds is not None:
            for hline in self.weight_thresholds:
                fig.add_shape(type="line", x0=0, x1=max(data["sorted"]), y0=hline, y1=hline, line=dict(color="black", width=1.5, dash="dash"))

        file_path = self.result_path / "weights_distribution.html"
        fig.write_html(str(file_path))
        return ReportOutput(path=file_path, name="weights distribution plot")