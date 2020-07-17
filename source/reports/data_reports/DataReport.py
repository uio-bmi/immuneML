from abc import ABC

from source.data_model.dataset.Dataset import Dataset
from source.reports.Report import Report


class DataReport(Report, ABC):

    def __init__(self, dataset: Dataset = None, result_path: str = None, name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path
        self.name = name
