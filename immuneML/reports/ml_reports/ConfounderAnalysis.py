from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class ConfounderAnalysis(MLReport):
    """
    ...
    """

    @classmethod
    def build_object(cls, **kwargs):
        # add checks?
        additional_labels = kwargs["additional_labels"]
        # print("add lbls ", additional_labels)
        # print("add lbls ", type(additional_labels))
        return ConfounderAnalysis(additional_labels=additional_labels)
        # return ConfounderAnalysis(**kwargs)

    def __init__(self, additional_labels: List[str], train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: MLMethod = None,
                 result_path: Path = None, name: str = None, hp_setting: HPSetting = None, label=None):
        super().__init__(train_dataset, test_dataset, method, result_path, name, hp_setting, label)

        self._additional_labels = additional_labels

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        paths = []

        # make predictions
        predictions = self.method.predict(self.test_dataset.encoded_data, self.label)  # label = disease

        # additional_labels = self.test_dataset.get_metadata(
        #     self._additional_labels)  # labels: ["signal_HLA", "signal_age"]
        # true_labels = {"age": additional_labels['signal_age'],
        #                self.label: self.test_dataset.encoded_data.labels[self.label],
        #                "hla": additional_labels['signal_HLA']}

        # true_labels = self.test_dataset.get_metadata(["signal_disease", "signal_HLA", "signal_age"])
        true_labels = self.test_dataset.get_metadata(["signal_disease", self._additional_labels])

        print((true_labels))

        # print("predictions done")
        # print(1*np.array(true_labels[self.label]))
        # print(type(true_labels[self.label]))
        # print(predictions["signal_disease"]*1)
        # print(type(predictions["signal_disease"]))
        # print("signal age", true_labels["signal_age"])
        print("ad labs ", type(self._additional_labels))

        for add_label in list([self._additional_labels]):
            print("label ", add_label)
            report_output_fig = self._metrics(add_label, predictions, true_labels)
            paths.append(report_output_fig)

        return ReportResult(name=self.name, output_figures=[ReportOutput(self.result_path / "report.html", "")])

    def _plot(self, plotting_data, output_name, metric, add_label):
        filename = self.result_path / f"{output_name}.html"

        figure = px.bar(plotting_data, x=f"{add_label}", y=f"{metric}", template='plotly_white',
                        title=f"{output_name}")
        figure.update_traces(marker_color=px.colors.sequential.Teal[3])

        figure.write_html(str(filename))

        return ReportOutput(filename)

    def _metrics(self, add_label, predictions, true_labels):
        # print("label ", add_label)
        # print("self.label ", self.label)

        for metric in ["FP", "FN"]:
            print("metric", metric)
            if metric == "FP":
                metric_inds = np.nonzero(np.greater(predictions[self.label], true_labels[self.label]))[0].tolist()
            else:
                metric_inds = np.nonzero(np.less(predictions[self.label], true_labels[self.label]))[0].tolist()
            print(f"{metric}_inds",metric_inds)
            label_inds = np.array(true_labels[add_label])[metric_inds]
            metric_age = np.count_nonzero(label_inds)
            # todo adapt later to non-binary confounders
            plotting_data_age = pd.DataFrame(
                {f"{metric}": [len(label_inds) - metric_age, metric_age], f"{add_label}": [False, True]})
            report_output_fig = self._plot(plotting_data=plotting_data_age, output_name=f"{metric}_{add_label}", metric=metric, add_label=add_label)

            return report_output_fig
