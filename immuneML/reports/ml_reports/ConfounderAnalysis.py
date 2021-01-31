from pathlib import Path
from typing import List
import plotly.express as px
import numpy as np
import pandas as pd

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
        # additional_labels = kwargs["additional_labels"]

        # return ConfounderAnalysis(additional_labels=additional_labels)
        return ConfounderAnalysis(**kwargs)

    def __init__(self, train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: MLMethod = None,
                 result_path: Path = None, name: str = None, hp_setting: HPSetting = None, label=None):
        super().__init__(train_dataset, test_dataset, method, result_path, name, hp_setting, label)

        # self._additional_labels = additional_labels

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
        true_labels = self.test_dataset.get_metadata(["signal_disease", "signal_HLA", "signal_age"])

        print("predictions done")
        print(true_labels[self.label])
        print(predictions["signal_disease"])
        print("signal age", true_labels["signal_age"])

        fp_inds = np.nonzero(np.greater(predictions[self.label], true_labels[self.label]))[0].tolist()
        print("fp_inds",fp_inds)
        age_inds = np.array(true_labels["signal_age"])[fp_inds]
        fp_age = np.count_nonzero(age_inds)
        # todo adapt later to non-binary confounders
        plotting_data_age = pd.DataFrame(
            {"fps": [len(age_inds) - fp_age, fp_age], "age": [False, True]})
        report_output_fig = self._plot(plotting_data=plotting_data_age, output_name="FP_age")
        paths.append(report_output_fig)

        fn_inds = np.nonzero(np.less(predictions[self.label], true_labels[self.label]))[0].tolist()
        age_inds = np.array(true_labels["signal_age"])[fn_inds]
        fn_age = np.count_nonzero(age_inds)
        # todo adapt later to non-binary confounders
        plotting_data_age = pd.DataFrame(
            {"fps": [len(age_inds) - fn_age, fn_age], "age": [False, True]})
        report_output_fig = self._plot(plotting_data=plotting_data_age, output_name="FN_age")
        paths.append(report_output_fig)

        return ReportResult(name=self.name, output_figures=[ReportOutput(self.result_path / "report.html", "")])

    def _plot(self, plotting_data, output_name):
        filename = self.result_path / f"{output_name}.html"

        figure = px.bar(plotting_data, x=f"age", y=f"fps", template='plotly_white',
                        title=f"{output_name}")
        figure.update_traces(marker_color=px.colors.sequential.Teal[3])

        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(filename)

    # def _confounder_on_metric(self, confounder, metric):
