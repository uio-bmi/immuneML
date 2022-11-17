import plotly.graph_objects as go

from pathlib import Path
from numpy import array
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.UnsupervisedMLReport import UnsupervisedMLReport
from immuneML.util.PathBuilder import PathBuilder
import numpy as np
import matplotlib.pylab as plt
plt.style.use("seaborn")

import pandas as pd
import plotly.graph_objs as go


class GeneratorReportPWM(UnsupervisedMLReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "GeneratorReport"
        return GeneratorReportPWM(name=name)

    def __init__(self, dataset: Dataset = None, method: UnsupervisedMLMethod = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.dataset = dataset

    def _generate(self) -> ReportResult:
        heatmap = self.result_path / f"{self.name}Heatmap.html"
        PathBuilder.build(self.result_path)

        fig = go.Figure(data=go.Heatmap(
            x=np.arange(1, len(self.method.model[0]) + 1),
            y=list(reversed(list(self.method.alphabet))),
            z=list(reversed(self.method.model.T,))))
        fig.update_xaxes(side="top", dtick=1)
        with heatmap.open("w") as file:
            fig.write_html(file)

        fullTable = self.result_path / f"{self.name}FullTable.html"

        HTMLTable = pd.DataFrame(self.method.model.T, index=self.method.alphabet, columns=np.arange(1, len(self.method.model.T[0]) + 1))
        with fullTable.open('w') as file:
            HTMLTable.to_html(file)

        result = ReportOutput(heatmap, name="Heatmap")
        tableResult = ReportOutput(fullTable, name="Raw table")

        generated_sequences = self.result_path / f"{self.name}GeneratedSequences.csv"

        data = pd.DataFrame(self.method.generated_sequences, columns=["Generated Sequences"])
        data.to_csv(generated_sequences, index=False)
        sequences_to_output = ReportOutput(generated_sequences, name="Generated Sequences")

        return ReportResult(self.name, output_figures=[tableResult, result], output_tables=[tableResult, sequences_to_output])
