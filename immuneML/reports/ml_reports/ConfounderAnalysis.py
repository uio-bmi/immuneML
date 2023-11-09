from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ConfounderAnalysis(MLReport):
    """
    A report that plots the numbers of false positives and false negatives with respect to each value of
    the metadata features specified by the user. This allows checking whether a given machine learning model makes more
    misclassifications for some values of a metadata feature than for the others.

    Specification arguments:

    - metadata_labels (list): A list of the metadata features to use as a basis for the calculations


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_confounder_report:
            ConfounderAnalysis:
                metadata_labels:
                  - age
                  - sex

    """

    @classmethod
    def build_object(cls, **kwargs):

        ParameterValidator.assert_keys(kwargs.keys(), ['metadata_labels', 'name'], ConfounderAnalysis.__name__, ConfounderAnalysis.__name__)
        ParameterValidator.assert_type_and_value(kwargs['metadata_labels'], list, ConfounderAnalysis.__name__, 'metadata_labels')
        ParameterValidator.assert_all_type_and_value(kwargs['metadata_labels'], str, ConfounderAnalysis.__name__, 'metadata_labels')
        ParameterValidator.assert_type_and_value(kwargs['name'], str, ConfounderAnalysis.__name__, 'name')

        return ConfounderAnalysis(metadata_labels=kwargs['metadata_labels'], name=kwargs['name'])

    def __init__(self, metadata_labels: List[str], train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: MLMethod = None, result_path: Path = None, name: str = None, hp_setting: HPSetting = None,
                 label=None, number_of_processes: int=1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method, result_path=result_path,
                         name=name, hp_setting=hp_setting, label=label, number_of_processes=number_of_processes)
        self.metadata_labels = metadata_labels

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        paths = []

        # make predictions
        predictions = self.method.predict(self.test_dataset.encoded_data, self.label)[self.label.name]

        true_labels = self.test_dataset.get_metadata(self.metadata_labels + [self.label.name])
        metrics = ["FP", "FN"]

        plot = make_subplots(rows=len(self.metadata_labels), cols=2)
        listOfPlot = []

        for label_index, meta_label in enumerate(self.metadata_labels):
            csv_data = {}
            for metric_index, metric in enumerate(metrics):
                plotting_data = self._metrics(metric=metric, label_name=self.label.name, meta_label=meta_label,
                                              predictions=predictions, true_labels=true_labels)

                csv_data[f"{metric}"] = plotting_data[f"{metric}"]

                plot.add_trace(go.Bar(x=plotting_data[meta_label], y=plotting_data[metric]), row=label_index + 1, col=metric_index + 1)
                plot.update_xaxes(title_text=f"{meta_label}", row=label_index + 1, col=metric_index + 1, type='category')
                plot.update_yaxes(title_text=f"{metric}", row=label_index + 1, col=metric_index + 1, rangemode="nonnegative", tick0=0, dtick=1)

            csv_data[f"{meta_label}"] = plotting_data[f"{meta_label}"]

            csv_data = pd.DataFrame(csv_data)

            listOfPlot.append(csv_data)

        plot.update_traces(marker_color=px.colors.sequential.Teal[3], showlegend=False)
        filename = self.result_path / "plots.html"
        plot.write_html(str(filename))
        report_output_fig = ReportOutput(filename)
        paths.append(report_output_fig)

        result_table_path = self._write_results_table(listOfPlot, self.metadata_labels)
        return ReportResult(name=self.name,
                            info="Plots the numbers of false positives and false negatives with respect to each value of the metadata features specified by the user.",
                            output_figures=paths, output_tables=[ReportOutput(result_table_path[0])])

    def _write_results_table(self, plotting_data, labels):
        filepaths = []
        for label_index, label in enumerate(labels):
            filepath = self.result_path / f"{label}.csv"
            plotting_data[label_index].to_csv(filepath, index=False)
            filepaths.append(filepath)
        return filepaths

    @staticmethod
    def _metrics(metric, label_name, meta_label, predictions, true_labels):
        # indices of samples at which misclassification occurred
        if metric == "FP":
            metric_inds = np.nonzero(np.greater(predictions, true_labels[label_name]))[0].tolist()
        else:
            metric_inds = np.nonzero(np.less(predictions, true_labels[label_name]))[0].tolist()

        metadata_values = true_labels[meta_label]
        # indices of misclassification with respect to the metadata label
        label_inds = np.array(metadata_values)[metric_inds]

        metric_vals = []
        unique_levels = np.unique(metadata_values)

        # number of metric occurrences at each metadata level
        for val in unique_levels:
            metric_vals.append(np.count_nonzero(label_inds == val))

        plotting_data = pd.DataFrame(
            {f"{metric}": metric_vals, f"{meta_label}": unique_levels})

        return plotting_data
