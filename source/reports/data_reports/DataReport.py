from abc import ABC

from source.data_model.dataset.Dataset import Dataset
from source.reports.Report import Report


class DataReport(Report, ABC):

    def __init__(self, dataset: Dataset = None, path: str = None):
        self.dataset = dataset
        self.path = path

    pass
