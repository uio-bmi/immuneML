import matplotlib.pyplot as plt
from pathlib import Path
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.LSTM import LSTM
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.GeneratorReport import GeneratorReport
from immuneML.util.PathBuilder import PathBuilder
import pandas as pd
import plotly.express as px
import logomaker


class GeneratorReportLSTM(GeneratorReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "GeneratorReportLSTM"
        return GeneratorReportLSTM(name=name)

    def __init__(self, dataset: Dataset = None, method: LSTM = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)

    def _generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)
        output_figures = []

        loss_over_time = self.result_path / f"{self.name}loss.html"
        generated_sequences = self.result_path / f"{self.name}GeneratedSequences.csv"
        logo_path = self.result_path / f"{self.name}Logo.png"
        data_logo_path = self.result_path / f"{self.name}Data_Logo.png"

        data = pd.DataFrame(enumerate(self.sequences), columns=["id", "sequence_aa"])
        data.to_csv(generated_sequences, index=False)

        self._make_generated_logo(logo_path)
        output_figures.append(ReportOutput(logo_path, name="Generated Logo"))

        if self.dataset:
            sequences = self.dataset.encoded_data.examples
            sequences_alpha = "".join([self.alphabet[i] for i in sequences]).split(" ")

            sequences_by_length = {}
            for sequence in sequences_alpha:
                if len(sequence) not in sequences_by_length:
                    sequences_by_length[len(sequence)] = [sequence]
                else:
                    sequences_by_length[len(sequence)].append(sequence)

            new_sequences = []
            for sequences in sequences_by_length.values():
                if len(sequences) > len(new_sequences):
                    new_sequences = sequences


            #new_sequences_alpha = [alphabet[c] for sequence in sequences for c in sequence].split(" ")
            data_counts = logomaker.alignment_to_matrix(sequences=new_sequences, to_type='counts')
            logomaker.Logo(data_counts, color_scheme="dmslogo_funcgroup")
            plt.grid(False)
            plt.savefig(data_logo_path)
            output_figures.append(ReportOutput(data_logo_path, name="Dataset Logo"))

        if self.method.historydf is not None:
            fig = px.line(self.method.historydf['data'][0])
            with loss_over_time.open("w", encoding="utf-8") as file:
                fig.write_html(file)
            output_figures.append(ReportOutput(loss_over_time, name="Loss"))

        sequences_to_output = ReportOutput(generated_sequences, name="Generated Sequences")

        return ReportResult(self.name, output_figures=output_figures, output_tables=[sequences_to_output])
