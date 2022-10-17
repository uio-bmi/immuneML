from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult
from immuneML.ml_methods.GenerativeModel import GenerativeModel


@dataclass
class GenerativeModelUnit:

    dataset: Dataset
    report: Report
    genModel: GenerativeModel
    encoder: DatasetEncoder = None
    label_config: LabelConfiguration = None
    report_result: ReportResult = None
    path: Path = None
    predictions_path: Path = None