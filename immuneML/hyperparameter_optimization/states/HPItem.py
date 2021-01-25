from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.reports.ReportResult import ReportResult


@dataclass
class HPItem:

    method: MLMethod = None
    encoder: DatasetEncoder = None
    performance: dict = None
    hp_setting: HPSetting = None
    train_predictions_path: Path = None
    test_predictions_path: Path = None
    ml_details_path: Path = None
    train_dataset: Dataset = None
    test_dataset: Dataset = None
    split_index: int = None
    model_report_results: List[ReportResult] = field(default_factory=list)
    encoding_train_results: List[ReportResult] = field(default_factory=list)
    encoding_test_results: List[ReportResult] = field(default_factory=list)
