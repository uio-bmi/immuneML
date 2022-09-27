from pathlib import Path

import plotly.express as px
import pandas as pd

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder

# todo maybe think of a better (more general) name if this is going to contain many motif reports
class PositionalMotifFrequencies(EncodingReport):
    """


    """

    @classmethod
    def build_object(cls, **kwargs):
        return PositionalMotifFrequencies(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)

        self.name = name

    def get_sequence_length(self):
        my_sequence = next(self.dataset.get_data())
        return len(my_sequence.get_sequence())

    def _generate(self):
        PathBuilder.build(self.result_path)

        motifs = self.dataset.encoded_data.feature_names
        plotting_data = self._get_plotting_data(motifs)

        report_output_fig = self._safe_plot(stacked_bar=plotting_data)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(name=self.name, output_figures=output_figures)

    def _get_plotting_data(self, motifs):
        positional_aa_counts = {aa: [0 for i in range(self.get_sequence_length())]
                                for aa in EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID)}

        for motif in motifs:
            indices, amino_acids = PositionalMotifHelper.string_to_motif(motif, "&", "-")

            for index, amino_acid in zip(indices, amino_acids):
                positional_aa_counts[amino_acid][index] += 1

        stacked_bar = pd.DataFrame(positional_aa_counts)
        stacked_bar = stacked_bar.loc[:, (stacked_bar != 0).any(axis=0)]

        return stacked_bar

    def _get_color_discrete_sequence(self):
        return px.colors.qualitative.Pastel[:-1] + px.colors.qualitative.Set3

    def _plot(self, stacked_bar) -> ReportOutput:
        file_path = self.result_path / f"positional_motif_frequencies.html"

        fig = px.bar(stacked_bar,
                     labels={"index": "Amino acid position", "value": "Frequency across high-scoring motifs"},
                     text="variable", color_discrete_sequence=self._get_color_discrete_sequence(), template='plotly_white')
        fig.update_layout(showlegend=False, font={"size": 14}, xaxis={"tickmode": "linear"})
        fig.write_html(str(file_path))

        return ReportOutput(path=file_path, name=f"Frequencies of amino acids found in the high-precision high-recall motifs")



# todo add tests under ImmuneML/test/reports/encoding_reports/test_<name_of_class>.py, see other tests in that folder as examples