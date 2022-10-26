from abc import ABC
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.reports.Report import Report


class UnsupervisedMLReport(Report, ABC):
    def __init__(self, dataset: Dataset = None, method: UnsupervisedMLMethod = None,
                 result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(name, number_of_processes)
        self.dataset = dataset
        self.method = method
        self.result_path = result_path
        self.name = name

    @staticmethod
    def get_title():
        return "Unsupervised ML model reports"
