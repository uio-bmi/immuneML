import abc
from pathlib import Path

from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState


class ClusteringReport(Report):

    DOCS_TITLE = "Clustering Instruction Reports"

    def __init__(self, name: str = None, result_path: Path = None, number_of_processes: int = 1,
                 state: ClusteringState = None):
        super().__init__(name, result_path, number_of_processes)
        self.state = state

    @classmethod
    def build_object(cls, **kwargs):
        pass

    @abc.abstractmethod
    def _generate(self) -> ReportResult:
        pass
