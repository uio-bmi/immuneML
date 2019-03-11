import abc

from source.data_model.dataset.Dataset import Dataset
from source.reports.Report import Report


class DataReport(Report):

    def generate_report(self, params):
        return self.generate(dataset=params["dataset"], result_path=params["result_path"], params=params)

    @abc.abstractmethod
    def generate(self, dataset: Dataset, result_path: str, params: dict):
        pass
