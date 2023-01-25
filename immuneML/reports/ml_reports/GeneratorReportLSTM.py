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
import pandas as pd
import plotly.express as px

import plotly.graph_objs as go


class GeneratorReportLSTM(UnsupervisedMLReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "GeneratorReportLSTM"
        return GeneratorReportLSTM(name=name)

    def __init__(self, dataset: Dataset = None, method: UnsupervisedMLMethod = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.dataset = dataset

    def _generate(self) -> ReportResult:

        loss_over_time = self.result_path / f"{self.name}loss.html"
        PathBuilder.build(self.result_path)

        fig = px.line(self.method.historydf['data'][0])
        with loss_over_time.open("w") as file:
            fig.write_html(file)

        generated_sequences = self.result_path / f"{self.name}GeneratedSequences.csv"

        data = pd.DataFrame(self.method.generated_sequences, columns=["Generated Sequences"])
        data.to_csv(generated_sequences, index=False)

        sequences_to_output = ReportOutput(generated_sequences, name="Generated Sequences")
        loss = ReportOutput(loss_over_time, name="Loss")

        return ReportResult(self.name, output_figures=[loss], output_tables=[sequences_to_output])
