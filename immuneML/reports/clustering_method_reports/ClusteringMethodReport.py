from abc import ABC
from pathlib import Path

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.reports.Report import Report


class ClusteringMethodReport(Report, ABC):
    """
    Clustering method reports show some features or statistics about the clustering method.
    """

    DOCS_TITLE = "Clustering method reports"

    def __init__(self, dataset: Dataset = None, model: ClusteringMethod = None, result_path: Path = None,
                 name: str = None):
        super().__init__(name=name, result_path=result_path)
        self.dataset = dataset
        self.model = model
