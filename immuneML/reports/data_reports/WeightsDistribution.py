import plotly.express as px
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper

class WeightsDistribution(DataReport):
    """

    """
    @classmethod
    def build_object(cls, **kwargs):
        return WeightsDistribution(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None, label: dict = None):
            super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
            self.label = label
            self.label_config = None

    def check_prerequisites(self):
        if self.dataset.get_example_weights() is not None:
            return True
        else:
            warnings.warn("WeightsDistribution: report requires weighting. Skipping this report...")

    def _generate(self) -> ReportResult:
        self.label_config = LabelHelper.create_label_config([self.label], self.dataset, WeightsDistribution.__name__,
                                                       f"{WeightsDistribution.__name__}/label")
        # weights = self.dataset.get_example_weights()
        # report_output_fig = self._safe_plot(weights=weights)
        data, color = self._get_plotting_data()
        report_output_fig = self._safe_plot(data=data, color=color)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(name=self.name,
                                info="A line graph of weights over sequences for is-binding and non-binding sequences",
                                output_figures=output_figures)



    def get_weights_by_class_df(self, data, weights):
        data["weights"] = weights
        data.sort_values(by=["weights"], inplace=True, ascending=False)

        data["sorted"] = None
        data.loc[data["is_binding"] == 0, "sorted"] = range(len(data[data["is_binding"] == 0]))
        data.loc[data["is_binding"] == 1, "sorted"] = range(len(data[data["is_binding"] == 1]))

        return data

    def get_weights_both_classes_df(self, data, weights):
        data["weights"] = weights
        data.sort_values(by=["weights"], inplace=True, ascending=False)
        data.loc[:, "sorted"] = range(len(data))

        return data

    def _get_plotting_data(self):
        weights = self.dataset.get_example_weights()
        data = self.dataset.get_metadata(["is_binding", "count"], return_df=True)
        if self.label == "is_binding":
            data = self.get_weights_by_class_df(data, weights)
            color = "is_binding"
        else:
            data = self.get_weights_both_classes_df(data, weights)
            color = None
        return data, color

    def _plot(self, data, color):
        test_hlines = [0.02, 0.3, 1]
        PathBuilder.build(self.result_path)
        fig = px.line(data, y="weights", color=color, x="sorted", log_y=True,
                      color_discrete_sequence=["#65D0B8", "#AB80D8"],
                      labels={
                          "weights": "Weights",
                          "sorted": "Sequences (sorted by weight)",
                          "is_binding": "Is binder"
                      }, template="plotly_white")

        for hline in test_hlines:
            fig.add_shape(type="line", x0=0, x1=max(data["sorted"]), y0=hline, y1=hline, line=dict(color="black", width=1.5, dash="dash"))
        # if hline is not None:
        #     fig.add_shape(type="line", x0=0, x1=max(data["sorted"]), y0=hline, y1=hline, line=dict(color="black", width=1.5, dash="dash"))

        # fig.update_layout(
        #     font=dict(
        #         size=fontsize,
        #     )
        # )
        file_path = self.result_path / "weights_distribution.html"
        fig.write_html(str(file_path))
        return ReportOutput(path=file_path, name="weights distribution plot")

    """
    def plot_raw_weight_scores_line(infile="../data/mason/train_val_test/mason_train.csv",
                                    split_classes=True, hline=1, fontsize=None):
        data = pd.read_csv(infile, sep=',')
        np_sequences, y_true, weights = read_data_file(infile, np.inf)

        if split_classes:
            data = get_weights_by_class_df(data.copy(), weights)
            color = "is_binding"
        else:
            data = get_weights_both_classes_df(data.copy(), weights)
            color = None


        fig = px.line(data, y="weights", color=color, x="sorted", log_y=True,
                      color_discrete_sequence=["#65D0B8", "#AB80D8"],
                      labels={
                          "weights": "Weights",
                          "sorted": "Sequences (sorted by weight)",
                          "is_binding": "Is binder"
                      }, template="plotly_white")

        if hline is not None:
            fig.add_shape(type="line", x0=0, x1=max(data["sorted"]), y0=hline, y1=hline, line=dict(color="black", width=1.5, dash="dash"))

        fig.update_layout(
            font=dict(
                size=fontsize,
            )
        )

        fig.show()
    """