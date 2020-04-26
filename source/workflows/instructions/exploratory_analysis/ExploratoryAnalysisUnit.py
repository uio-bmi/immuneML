from dataclasses import dataclass

from source.data_model.dataset.Dataset import Dataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.environment.LabelConfiguration import LabelConfiguration
from source.reports.Report import Report
from source.reports.ReportResult import ReportResult


@dataclass
class ExploratoryAnalysisUnit:
    dataset: Dataset
    report: Report
    preprocessing_sequence: list = None
    encoder: DatasetEncoder = None
    label_config: LabelConfiguration = None
    batch_size: int = 1
    report_result: ReportResult = None
