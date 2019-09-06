from abc import ABC

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.ml_methods.MLMethod import MLMethod
from source.reports.Report import Report


class MLReport(Report, ABC):

    def __init__(self, train_dataset: RepertoireDataset = None, test_dataset: RepertoireDataset = None, method: MLMethod = None,
                 result_path: str = None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.method = method
        self.result_path = result_path
