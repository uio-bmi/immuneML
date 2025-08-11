from abc import ABC
from pathlib import Path

from immuneML.reports.Report import Report
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


class ClusteringMethodReport(Report, ABC):
    """
    Clustering method reports show some features or statistics about the clustering method.
    """

    DOCS_TITLE = "Clustering method reports"

    def __init__(self, result_path: Path = None, name: str = None, clustering_item: ClusteringItem = None):
        super().__init__(name=name, result_path=result_path)
        self.item = clustering_item
