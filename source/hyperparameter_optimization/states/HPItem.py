from dataclasses import dataclass, field
from typing import List

from source.data_model.dataset.Dataset import Dataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.ml_methods.MLMethod import MLMethod
from source.reports.ReportResult import ReportResult


@dataclass
class HPItem:

    method: MLMethod = None
    encoder: DatasetEncoder = None
    performance: dict = None
    hp_setting: HPSetting = None
    train_predictions_path: str = None
    test_predictions_path: str = None
    ml_details_path: str = None
    train_dataset: Dataset = None
    test_dataset: Dataset = None
    split_index: int = None
    model_report_results: List[ReportResult] = field(default_factory=list)
    encoding_train_results: List[ReportResult] = field(default_factory=list)
    encoding_test_results: List[ReportResult] = field(default_factory=list)
