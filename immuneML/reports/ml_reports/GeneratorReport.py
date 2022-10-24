from pathlib import Path
from numpy import array
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder

import plotly.graph_objs as go


class GeneratorReport(MLReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "GeneratorReport"
        return GeneratorReport(name=name)

    def __init__(self, dataset: Dataset = None, train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: MLMethod = None, result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.dataset = dataset

    def _generate(self) -> ReportResult:

        generatorReport = ReportResult()

        return ReportResult()
