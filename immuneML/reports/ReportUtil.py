import copy
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.reports.ml_reports.MLReport import MLReport


class ReportUtil:

    @staticmethod
    def _make_new_report(report: Report, path: Path, context: dict):
        tmp_report = copy.deepcopy(report)
        report_name = report.name if report.name is not None else 'report_result'
        tmp_report.result_path = path / report_name
        tmp_report.set_context(context)
        return tmp_report

    @staticmethod
    def run_ML_reports(train_dataset: Dataset, test_dataset: Dataset, method: MLMethod, reports: List[MLReport], path: Path,
                       hp_setting: HPSetting, label: str, context: dict = None) -> List[ReportResult]:
        report_results = []
        for report in reports:
            tmp_report = ReportUtil._make_new_report(report, path, context)
            tmp_report.method = method
            tmp_report.train_dataset = train_dataset
            tmp_report.test_dataset = test_dataset
            tmp_report.hp_setting = hp_setting
            tmp_report.label = label
            result = tmp_report.generate_report()
            report_results.append(result)
        return report_results

    @staticmethod
    def _run_reports_on_dataset(dataset: Dataset, reports: list, path: Path, context: dict = None) -> List[ReportResult]:
        report_results = []
        for report in reports:
            tmp_report = ReportUtil._make_new_report(report, path, context)
            tmp_report.dataset = dataset
            result = tmp_report.generate_report()
            report_results.append(result)
        return report_results

    @staticmethod
    def run_encoding_reports(dataset: Dataset, reports: List[EncodingReport], path: Path, context: dict = None) -> List[ReportResult]:
        return ReportUtil._run_reports_on_dataset(dataset, reports, path, context)

    @staticmethod
    def run_data_reports(dataset: Dataset, reports: List[DataReport], path: Path, context: dict = None):
        return ReportUtil._run_reports_on_dataset(dataset, reports, path, context)
