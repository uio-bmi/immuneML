import plotly.graph_objects as go

from pathlib import Path
from numpy import array
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use("seaborn")

import plotly.graph_objs as go


class GeneratorReport(MLReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "GeneratorReport"
        return GeneratorReport(name=name)

    def __init__(self, dataset: Dataset = None, train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: UnsupervisedMLMethod = None, result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.dataset = dataset

    def _generate(self) -> ReportResult:

        fig = go.Figure(data=go.Heatmap(
            z=self.method.model,
            x=self.method.alphabet))
        fig.show()

        return ReportResult()
