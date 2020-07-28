from abc import ABC

from source.data_model.dataset.Dataset import Dataset
from source.ml_methods.MLMethod import MLMethod
from source.reports.Report import Report


class MLReport(Report, ABC):

    def __init__(self, train_dataset: Dataset = None, test_dataset: Dataset = None, method: MLMethod = None,
                 result_path: str = None, name: str = None):
        super().__init__(name)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.method = method
        self.result_path = result_path
        self.name = name
