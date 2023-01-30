import abc
import warnings
from pathlib import Path

from immuneML.analysis.data_manipulation.DataReshaper import DataReshaper
from immuneML.data_model.dataset.Dataset import Dataset
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
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.x = "feature"
        self.color = color_grouping_label
        self.facet_row = row_grouping_label
        self.facet_column = column_grouping_label

    def _generate_report_result(self) -> ReportResult:
        PathBuilder.build(self.result_path)
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

    @abc.abstractmethod
    def _plot(self, data_long_format) -> ReportOutput:
        pass

    def check_prerequisites(self):
        location = self.__class__.__name__
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(
                f"{location}: this report can only be created for an encoded dataset. {location} report will not be created.")
            run_report = False
        elif len(self.dataset.encoded_data.examples.shape) != 2:
            warnings.warn(
                f"{location}: this report can only be created for a 2-dimensional encoded dataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.get_label_names())

            labels = [self.color, self.facet_row, self.facet_column]

            for label_param in labels:
                if label_param is not None:
                    if label_param not in legal_labels:
                        warnings.warn(
                            f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

        return run_report
