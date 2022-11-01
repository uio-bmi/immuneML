from pathlib import Path

import plotly.express as px
import pandas as pd

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import (
    PositionalMotifHelper,
)
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class PositionalMotifFrequencies(EncodingReport):
    """
    Plots a stacked bar plot of amino acid occurrence at different indices in any given dataset, along with a plot
    investigating motif continuity which displays a bar plot of the gap sizes between the amino acids in the motifs in
    the given dataset. Note that a distance of 1 means that the amino acids are continuous (next to each other).

    YAML specification example:

    .. indent with spaces
    .. code-block:: yaml

      my_expl_analysis_instruction:
          type: ExploratoryAnalysis
          analyses:
              my_second_analysis:
                  dataset: d1
                  encoding: e1
                  report: r2
                  labels:
                      - is_binding
          number_of_processes: 4
    """

    @classmethod
    def build_object(cls, **kwargs):
        return PositionalMotifFrequencies(**kwargs)

    def __init__(
        self,
        dataset: RepertoireDataset = None,
        result_path: Path = None,
        name: str = None,
        number_of_processes: int = 1,
    ):
        super().__init__(
            dataset=dataset,
            result_path=result_path,
            name=name,
            number_of_processes=number_of_processes,
        )

        self.name = name

    def get_sequence_length(self):
        my_sequence = next(self.dataset.get_data())
        return len(my_sequence.get_sequence())

    def _generate(self):
        PathBuilder.build(self.result_path)

        motifs = self.dataset.encoded_data.feature_names
        positional_aa_counts_df = self._get_positional_aa_counts(motifs)
        gap_size_dict = self._get_gap_sizes(motifs)

        positional_aa_counts_table = self._write_positional_aa_counts_table(
            positional_aa_counts_df
        )
        gap_size_tables = self._write_gap_size_table(gap_size_dict)

        output_figures = self._safe_plot(
            positional_aa_counts_df=positional_aa_counts_df, gap_size_dict=gap_size_dict
        )
        return ReportResult(
            name=self.name,
            output_figures=output_figures,
            output_tables=gap_size_tables + [positional_aa_counts_table],
        )

    def _get_gap_sizes(self, motifs):
        gap_size_count = {
            motif_size: {
                gap_size: 0 for gap_size in range(1, self.get_sequence_length())
            }
            for motif_size in range(self.get_sequence_length())
        }

        for motif in motifs:
            indices, amino_acids = PositionalMotifHelper.string_to_motif(
                motif, "&", "-"
            )
            indices_gap_size = len(indices)

            for i in range(indices_gap_size - 1):
                gap_size_count[indices_gap_size][indices[i + 1] - indices[i]] += 1
        gap_size_dict = dict()

        for gap_size in gap_size_count:
            if sum(gap_size_count[gap_size].values()) != 0:
                gap_size_dict[gap_size] = gap_size_count[gap_size]

        return gap_size_dict

    def _get_positional_aa_counts(self, motifs):
        positional_aa_counts = {
            aa: [0 for i in range(self.get_sequence_length())]
            for aa in EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID)
        }

        for motif in motifs:
            indices, amino_acids = PositionalMotifHelper.string_to_motif(
                motif, "&", "-"
            )

            for index, amino_acid in zip(indices, amino_acids):
                positional_aa_counts[amino_acid][index] += 1

        positional_aa_counts_df = pd.DataFrame(positional_aa_counts)
        positional_aa_counts_df = positional_aa_counts_df.loc[
            :, (positional_aa_counts_df != 0).any(axis=0)
        ]

        return positional_aa_counts_df

    def _plot_gap_sizes(self, gap_size_dict):
        file_path = self.result_path / f"positional_motif_frequencies.html"
        gap_size_figs = []

        for motif_size in gap_size_dict:
            data = gap_size_dict[motif_size]
            file_path = self.result_path / f"gap_size_for_motif_size_{motif_size}.html"
            gap_size_fig = px.bar(
                x=data.keys(),
                y=data.values(),
                labels={
                    "x": "Distance between Amino Acids",
                    "y": "Distance occurrence",
                },
                color_discrete_sequence=self._get_color_discrete_sequence(),
                template="plotly_white",
                title=f"Distances between Amino Acids for {motif_size} positions",
            )
            gap_size_fig.update_layout(
                showlegend=False, font={"size": 14}, xaxis={"tickmode": "linear"}
            )
            gap_size_fig.write_html(str(file_path))

            gap_size_figs.append(
                ReportOutput(
                    path=file_path,
                    name=f"Gap size between amino acids in high-precision high-recall motifs",
                )
            )

        return gap_size_figs

    def _plot_positional_aa_counts(self, positional_aa_counts_df):
        file_path = self.result_path / f"positional_motif_frequencies.html"
        positional_aa_counts_fig = px.bar(
            positional_aa_counts_df,
            labels={
                "index": "Amino acid position",
                "value": "Frequency across high-scoring motifs",
            },
            text="variable",
            color_discrete_sequence=self._get_color_discrete_sequence(),
            template="plotly_white",
        )
        positional_aa_counts_fig.update_layout(
            showlegend=False, font={"size": 14}, xaxis={"tickmode": "linear"}
        )
        positional_aa_counts_fig.write_html(str(file_path))
        return ReportOutput(
            path=file_path,
            name=f"Frequencies of amino acids found in the high-precision high-recall motifs",
        )

    def _get_color_discrete_sequence(self):
        return px.colors.qualitative.Pastel[:-1] + px.colors.qualitative.Set3

    def _plot(self, positional_aa_counts_df, gap_size_dict) -> ReportOutput:
        report_outputs = self._plot_gap_sizes(gap_size_dict)
        report_outputs.append(self._plot_positional_aa_counts(positional_aa_counts_df))

        return report_outputs if len(report_outputs) != 0 else None

    def _write_positional_aa_counts_table(
        self, positional_aa_counts_df
    ) -> ReportOutput:
        table_path = self.result_path / f"positional_aa_counts.csv"
        positional_aa_counts_df.to_csv(table_path, index=True)
        return ReportOutput(
            path=table_path,
            name=f"Frequencies of amino acids found in the high-precision high-recall motifs",
        )

    def _write_gap_size_table(self, gap_size_dict) -> ReportOutput:
        gap_size_tables = []
        for motif_size in gap_size_dict:
            table_path = (
                self.result_path / f"gap_size_table_motif_size_{motif_size}.csv"
            )
            gap_size_sub_dict = gap_size_dict[motif_size]
            gap_size_df = pd.DataFrame.from_dict(
                gap_size_sub_dict, orient="index", dtype=str
            )
            gap_size_df.to_csv(table_path, index=True, header=["Gap size, occurance"])
            gap_size_tables.append(
                ReportOutput(
                    path=table_path,
                    name=f"Gap size between amino acids in high-precision high-recall motifs for motif size {motif_size}",
                )
            )

        return gap_size_tables
