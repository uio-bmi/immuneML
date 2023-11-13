import abc

from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult


class ClusteringReport(Report):
    @classmethod
    def build_object(cls, **kwargs):
        pass

    @abc.abstractmethod
    def _generate(self) -> ReportResult:
        pass
