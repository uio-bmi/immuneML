from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult


@dataclass
class GenerativeModelUnit:

    dataset: Dataset
    report: Report
    hp_setting: HPSetting
    label_config: LabelConfiguration
    pool_size: int
    name: str
    preprocessing_sequence: list = None
    encoder: DatasetEncoder = None
    label_config: LabelConfiguration = None
    report_result: ReportResult = None
    path: Path = None
    predictions_path: Path = None