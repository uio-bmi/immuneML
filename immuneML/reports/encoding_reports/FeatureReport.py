import abc
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from immuneML.analysis.data_manipulation.DataReshaper import DataReshaper
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class FeatureReport(EncodingReport):
    """
    Base class for reports that plot something about the reshaped feature values of any dataset.
    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None,
                 color_grouping_label: str = None, row_grouping_label=None, column_grouping_label=None,
                 name: str = None, number_of_processes: int = 1, error_function: str = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.x = "feature"
        self.color = color_grouping_label
        self.facet_row = row_grouping_label
        self.facet_column = column_grouping_label
        self.error_function = error_function

    def _generate_report_result(self, data_long_format=None) -> ReportResult:
        PathBuilder.build(self.result_path)
        if data_long_format is None:
            data_long_format = DataReshaper.reshape(self.dataset, self.dataset.get_label_names())
        table_result = self._write_results_table(data_long_format)
        report_output = self._safe_plot(data_long_format=data_long_format)
        output_tables = [table_result]
        if report_output is None:
            output_figures = None
        elif isinstance(report_output, tuple):
            output_figures = report_output[0] if isinstance(report_output[0], list) else [report_output[0]]
            output_tables = report_output[1]
        else:
            output_figures = report_output if isinstance(report_output, list) else [report_output]
        return ReportResult(name=self.name, output_figures=output_figures, output_tables=output_tables)

    def _write_results_table(self, data) -> ReportOutput:
        table_path = self.result_path / f"feature_values.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "feature values")

    def std(self, x):
        return x.std(ddof=0)

    def sem(self, x):
        return x.std(ddof=0) / (len(x) ** 0.5)

    @abc.abstractmethod
    def _plot(self, data_long_format) -> ReportOutput:
        pass

    def _get_error_function(self):
        if self.error_function == 'std':
            return self.std
        elif self.error_function == 'sem':
            return self.sem
        else:
            logging.warning(f"{self.__class__.__name__}: unknown error function '{self.error_function}'. "
                            f"Using 'std' as default.")
            return self.std

    def _get_grouped_data(self, data_long_format, cols_list):
        error_function = self._get_error_function()
        cols_list = [i for i in cols_list if i]
        cols_list = list(set(cols_list))
        grouped_data = data_long_format.groupby(cols_list, as_index=False).agg(
            {"value": ['mean', error_function]})
        grouped_data.columns = grouped_data.columns.map(''.join)
        return grouped_data

    def _summarise_feature_values(self, group_cols: List[str]) -> pd.DataFrame:
        """Mean and error per feature, computed directly from the (possibly sparse) encoded matrix, so that the
        example by feature matrix is never expanded to long format."""
        matrix = self.dataset.encoded_data.examples
        feature_names = self.dataset.encoded_data.feature_names
        feature_names = list(feature_names) if feature_names is not None else list(range(matrix.shape[1]))

        if len(group_cols) == 0:
            groups = {(): np.arange(matrix.shape[0])}
        else:
            metadata = pd.DataFrame(self.dataset.get_metadata(group_cols, return_df=True))
            groups = {key if isinstance(key, tuple) else (key,): np.asarray(indices)
                      for key, indices in metadata.groupby(group_cols, sort=False).indices.items()}

        annotations = self.dataset.encoded_data.feature_annotations
        if annotations is not None:
            annotations = annotations.reset_index(drop=True).drop(columns=[self.x], errors='ignore')

        error_name = self._get_error_function().__name__
        summaries = []
        for group_values, row_indices in groups.items():
            mean, error = self._mean_and_error(matrix[row_indices], error_name)
            summary = pd.DataFrame({self.x: feature_names, "valuemean": mean, f"value{error_name}": error})
            for col, value in zip(group_cols, group_values):
                summary[col] = value
            # annotations are in the same order as the features, so they are added by position: feature names
            # are not necessarily unique and joining on them could duplicate rows
            summaries.append(summary if annotations is None else pd.concat([summary, annotations], axis=1))

        return pd.concat(summaries, ignore_index=True)

    def _mean_and_error(self, matrix, error_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if issparse(matrix):
            mean = np.asarray(matrix.mean(axis=0)).ravel()
            mean_of_squares = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        else:
            matrix = np.asarray(matrix)
            mean = matrix.mean(axis=0)
            mean_of_squares = np.square(matrix).mean(axis=0)

        # population variance, to match the ddof=0 standard deviation used by FeatureReport
        error = np.sqrt(np.clip(mean_of_squares - np.square(mean), 0, None))
        if error_name == "sem":
            error = error / (matrix.shape[0] ** 0.5)

        return mean, error

    def check_prerequisites(self):
        location = self.__class__.__name__
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            logging.warning(
                f"{location}: this report can only be created for an encoded dataset. {location} report will not be created.")
            run_report = False
        elif len(self.dataset.encoded_data.examples.shape) != 2:
            logging.warning(
                f"{location}: this report can only be created for a 2-dimensional encoded dataset. {location} report will not be created.")
            run_report = False
        else:
            # labels the data was encoded with (e.g. `labels` of an ExploratoryAnalysis unit) can be used as well,
            # including gene labels such as j_call, which the dataset itself does not list as prediction targets
            encoded_labels = list(self.dataset.encoded_data.labels.keys()) if self.dataset.encoded_data.labels else []
            legal_labels = list(self.dataset.get_label_names(include_gene_labels=True)) + encoded_labels

            labels = [self.color, self.facet_row, self.facet_column]

            for label_param in labels:
                if label_param is not None:
                    if label_param not in legal_labels:
                        logging.warning(
                            f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

        return run_report
