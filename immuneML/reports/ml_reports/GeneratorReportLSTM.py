import matplotlib.pyplot as plt
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
import logomaker

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


        PathBuilder.build(self.result_path)

        loss_over_time = self.result_path / f"{self.name}loss.html"
        generated_sequences = self.result_path / f"{self.name}GeneratedSequences.csv"
        logo_path = self.result_path / f"{self.name}Logo.png"
        data_logo_path = self.result_path / f"{self.name}Data_Logo.png"

        data = pd.DataFrame(self.method.generated_sequences, columns=["Generated Sequences"])
        data.to_csv(generated_sequences, index=False)

        sequences_dict = {}
        output_figures = []

        for sequence in self.method.generated_sequences:
            if len(sequence) not in sequences_dict.keys():
                sequences_dict[len(sequence)] = [sequence]
            else:
                sequences_dict[len(sequence)].append(sequence)

        new_sequences = []
        for key in sequences_dict:
            if len(sequences_dict[key]) > len(new_sequences):
                new_sequences = sequences_dict[key]

        data_counts = logomaker.alignment_to_matrix(sequences=new_sequences, to_type='counts')
        logo = logomaker.Logo(data_counts, color_scheme="dmslogo_funcgroup")
        plt.grid(False)
        plt.savefig(logo_path)

        output_figures.append(ReportOutput(logo_path, name="Logo"))
        sequences_to_output = ReportOutput(generated_sequences, name="Generated Sequences")

        if self.dataset:
            sequences = self.dataset.encoded_data.examples.split(" ")
            sequences_dict = {}

            for sequence in sequences:
                if len(sequence) not in sequences_dict.keys():
                    sequences_dict[len(sequence)] = [sequence]
                else:
                    sequences_dict[len(sequence)].append(sequence)

            new_sequences = []
            for key in sequences_dict:
                if len(sequences_dict[key]) > len(new_sequences):
                    new_sequences = sequences_dict[key]

            data_counts = logomaker.alignment_to_matrix(sequences=new_sequences, to_type='counts')
            data_logo = logomaker.Logo(data_counts, color_scheme="dmslogo_funcgroup")
            plt.grid(False)
            plt.savefig(data_logo_path)
            output_figures.append(ReportOutput(data_logo_path, name="Dataset Logo"))

        if self.method.historydf is not None:
            fig = px.line(self.method.historydf['data'][0])
            with loss_over_time.open("w", encoding="utf-8") as file:
                fig.write_html(file)
            output_figures.append(ReportOutput(loss_over_time, name="Loss"))



        return ReportResult(self.name, output_figures=output_figures, output_tables=[sequences_to_output])