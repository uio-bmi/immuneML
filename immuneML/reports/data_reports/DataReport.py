from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.Report import Report


class DataReport(Report):
    """
    Data reports show some type of features or statistics about a given dataset.

    When running the :ref:`TrainMLModel` instruction, data reports can be specified under the key 'data_reports', to run the
    report on the whole dataset, or inside the 'selection' or 'assessment' specification under the keys 'reports/data' (current cross-validation split) or 'reports/data_splits' (train/test sub-splits).

    Alternatively, when running the :ref:`ExploratoryAnalysis` instruction, data reports can be specified under 'reports'.
    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path

    @staticmethod
    def get_title():
        return "Data reports"
