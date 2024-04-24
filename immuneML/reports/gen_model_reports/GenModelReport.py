from abc import ABC
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.Report import Report


class GenModelReport(Report, ABC):

    def __init__(self, dataset: Dataset = None, model: GenerativeModel = None, result_path: Path = None,
                 name: str = None):
        super().__init__(name=name, result_path=result_path)
        self.dataset = dataset
        self.model = model

    @staticmethod
    def get_title():
        return "Generative model reports"
