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

class PositionalMotifFrequencies(EncodingReport):
    """
    Plots a stacked bar plot of amino acid occurrence at different indices in any given dataset, along with a plot
    investigating motif continuity which displays a bar plot of the distances between the amino acids in the motifs in
    the given dataset. Note that a distance of 1 means that the amino acids are continuous (next to each other).

    # todo write args, YAML specification and other information that might be needed

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
        stacked_bar, hist_dict = self._get_plotting_data(motifs)

        output_figures = self._safe_plot(stacked_bar=stacked_bar, hist_dict=hist_dict)
        return ReportResult(name=self.name, output_figures=output_figures)


    def _get_plotting_data(self, motifs):
        positional_aa_counts = {aa: [0 for i in range(self.get_sequence_length())]
                                for aa in EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID)}

        distance_aa_counts = {position: {distance: 0 for distance in range(1, self.get_sequence_length())} for position in range(self.get_sequence_length())}

        for motif in motifs:
            indices, amino_acids = PositionalMotifHelper.string_to_motif(motif, "&", "-")
            distance = len(indices)

            for i in range(distance-1):
                distance_aa_counts[distance][indices[i+1]-indices[i]] += 1

            for index, amino_acid in zip(indices, amino_acids):
                positional_aa_counts[amino_acid][index] += 1

        hist_dict = dict()

        for distance_count in distance_aa_counts:
            if sum(distance_aa_counts[distance_count].values()) != 0:
                hist_dict[distance_count] = distance_aa_counts[distance_count]

        stacked_bar = pd.DataFrame(positional_aa_counts)
        stacked_bar = stacked_bar.loc[:, (stacked_bar != 0).any(axis=0)]


        return stacked_bar, hist_dict

    def _get_color_discrete_sequence(self):
        return px.colors.qualitative.Pastel[:-1] + px.colors.qualitative.Set3

    def _plot(self, stacked_bar, hist_dict) -> ReportOutput:
        file_path = self.result_path / f"positional_motif_frequencies.html"
        ReportOutputs = []
        stacked_bar_fig = px.bar(stacked_bar,
                     labels={"index": "Amino acid position", "value": "Frequency across high-scoring motifs"},
                     text="variable", color_discrete_sequence=self._get_color_discrete_sequence(), template='plotly_white')
        stacked_bar_fig.update_layout(showlegend=False, font={"size": 14}, xaxis={"tickmode": "linear"})
        stacked_bar_fig.write_html(str(file_path))
        ReportOutputs.append(ReportOutput(path=file_path, name=f"Frequencies of amino acids found in the high-precision high-recall motifs"))

        for position in hist_dict:
            data = hist_dict[position]
            file_path = self.result_path / f"motif_continuity_{position}_positions.html"
            hist_fig = px.bar(x=data.keys(), y=data.values(), labels={"x":"Distance between Amino Acids", "y":"Distance occurrence"}, color_discrete_sequence=self._get_color_discrete_sequence(), template='plotly_white', title=f"Distances between Amino Acids for {position} positions")
            hist_fig.update_layout(showlegend=False, font={"size": 14}, xaxis={"tickmode": "linear"})
            hist_fig.write_html(str(file_path))

            ReportOutputs.append(ReportOutput(path=file_path, name=f"Distance between amino acids in high-precision high-recall motifs"))

        return ReportOutputs


# todo add tests under ImmuneML/test/reports/encoding_reports/test_<name_of_class>.py, see other tests in that folder as examples