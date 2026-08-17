import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.reports.ReportResult import ReportResult


@dataclass
class HPItem:
    """
    Note on `method`/`_method`: nested CV can retain thousands of HPItems for the lifetime of a
    TrainMLModel run (one per inner-CV fold x hyperparameter setting), each holding a fully fitted model.
    By the time an HPItem is built, that model has already been used for prediction/reporting and already
    stored to disk (MLMethodTrainer stores it before MLProcess ever constructs an HPItem, recording where
    at `method.result_path`), so keeping it resident is often unnecessary. For methods that flag their
    fitted state as expensive to hold onto (MLMethod.uses_offloadable_resources(), e.g. a GPU-resident
    XGBoost model), `method` is offloaded right after construction and transparently reloaded from disk
    the next time `.method` is accessed - callers keep using `hp_item.method` exactly as before. For all
    other (typically lightweight, CPU-based) methods, nothing changes.
    """

    _method: MLMethod = None
    encoder: DatasetEncoder = None
    performance: dict = None
    hp_setting: HPSetting = None
    train_predictions_path: Path = None
    test_predictions_path: Path = None
    ml_settings_export_path: Path = None
    train_dataset: Dataset = None
    test_dataset: Dataset = None
    split_index: int = None
    model_report_results: List[ReportResult] = field(default_factory=list)
    encoding_train_results: List[ReportResult] = field(default_factory=list)
    encoding_test_results: List[ReportResult] = field(default_factory=list)

    # defined explicitly (dataclass then skips generating its own __init__) so that callers can keep
    # constructing HPItem(method=..., ...) - `method` is not itself a dataclass field, it is stored into
    # `_method` and exposed via the `method` property below.
    def __init__(self, method: MLMethod = None, encoder: DatasetEncoder = None, performance: dict = None,
                 hp_setting: HPSetting = None, train_predictions_path: Path = None, test_predictions_path: Path = None,
                 ml_settings_export_path: Path = None, train_dataset: Dataset = None, test_dataset: Dataset = None,
                 split_index: int = None, model_report_results: List[ReportResult] = None,
                 encoding_train_results: List[ReportResult] = None, encoding_test_results: List[ReportResult] = None):
        self._method = method
        self.encoder = encoder
        self.performance = performance
        self.hp_setting = hp_setting
        self.train_predictions_path = train_predictions_path
        self.test_predictions_path = test_predictions_path
        self.ml_settings_export_path = ml_settings_export_path
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split_index = split_index
        self.model_report_results = model_report_results if model_report_results is not None else []
        self.encoding_train_results = encoding_train_results if encoding_train_results is not None else []
        self.encoding_test_results = encoding_test_results if encoding_test_results is not None else []
        self._offload_method()

    @property
    def method(self) -> MLMethod:
        if self._method is not None and getattr(self._method, 'model', None) is None \
                and self._method.result_path is not None:
            try:
                self._method.load(self._method.result_path)
            except Exception as e:
                logging.warning(f"HPItem: could not reload previously offloaded model from "
                                f"{self._method.result_path} ({e}).")
        return self._method

    @method.setter
    def method(self, value: MLMethod):
        self._method = value

    def _offload_method(self):
        """Drops the heavy native model (e.g. a GPU-resident XGBoost Booster) once it has already been
        stored to disk and flagged as offloadable, so this HPItem can be kept around (e.g. for
        hyperparameter selection/reporting) without pinning that memory for the rest of the run."""
        if self._method is not None and self._method.result_path is not None \
                and getattr(self._method, 'model', None) is not None \
                and self._method.uses_offloadable_resources():
            self._method.model = None
            gc.collect()
