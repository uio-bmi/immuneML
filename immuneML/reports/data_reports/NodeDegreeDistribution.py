import logging
from pathlib import Path

from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class NodeDegreeDistribution(DataReport):

    def build_object(cls, **kwargs):
        #TODO: Add validation of kwargs
        return NodeDegreeDistribution(**kwargs)

    def __init__(self, dataset: SequenceDataset = None, result_path: Path = None, name: str = None,
                 compairr_path: str = None, indels: bool = False, threads: int = 4):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.compairr_path = compairr_path
        self.indels = indels
        self.threads = threads

    def check_prerequisites(self) -> bool:
        if not self.compairr_path:
            logging.warning("CompAIRR path not provided. CompAIRR must be installed and available in the system PATH.")
            return False
        if not isinstance(self.dataset, SequenceDataset):
            logging.warning(f"{NodeDegreeDistribution.__name__} report can only be generated for SequenceDataset.")
        return False

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        return ReportResult()
