from pathlib import Path

import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List
from plotly.subplots import make_subplots

from immuneML.data_model.dataset import SequenceDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import (
    PositionalMotifHelper,
)
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class PositionalMotifFrequencies(EncodingReport):
    """
    This report must be used in combination with the :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`.
    Plots a stacked bar plot of amino acid occurrence at different indices in any given dataset, along with a plot
    investigating motif continuity which displays a bar plot of the gap sizes between the amino acids in the motifs in
    the given dataset. Note that a distance of 1 means that the amino acids are continuous (next to each other).

    YAML specification example:

    .. indent with spaces
    .. code-block:: yaml

        my_pos_motif_report: PositionalMotifFrequencies

    """

    @classmethod
    def build_object(cls, **kwargs):
        return PositionalMotifFrequencies(**kwargs)

    def __init__(
        self,
        dataset: SequenceDataset = None,
        result_path: Path = None,
        name: str = None,
        number_of_processes: int = 1,
        max_gap_size_only: bool = None,
    ):
        super().__init__(
            dataset=dataset,
            result_path=result_path,
            name=name,
            number_of_processes=number_of_processes,
        )
        self.max_gap_size_only = max_gap_size_only

    def get_sequence_length(self):
        my_sequence = next(self.dataset.get_data())
        return len(my_sequence.get_sequence())

    def _generate(self):
        PathBuilder.build(self.result_path)

        motifs = self.dataset.encoded_data.feature_names
        positional_aa_counts_df = self._get_positional_aa_counts(motifs)
        gap_size_df = self._get_gap_sizes(motifs)

        positional_aa_counts_table = self._write_positional_aa_counts_table(
            positional_aa_counts_df
        )
        gap_size_tables = self._write_gap_size_table(gap_size_df)

        output_figures = self._safe_plot(
            positional_aa_counts_df=positional_aa_counts_df, gap_size_df=gap_size_df
        )
        return ReportResult(
            name=self.name,
            output_figures=output_figures,
            output_tables=[gap_size_tables, positional_aa_counts_table],
        )

    def _get_gap_sizes(self, motifs):
        gap_size_count = {
            motif_size: {
                gap_size: 0 for gap_size in range(0, self.get_sequence_length() - 1)
            }
            for motif_size in range(self.get_sequence_length())
        }

        for motif in motifs:
            indices, amino_acids = PositionalMotifHelper.string_to_motif(
                motif, "&", "-"
            )
            motif_size = len(indices)

            if self.max_gap_size_only and motif_size-1 != 0:
                gap_size = max([indices[i+1]-indices[i] -1 for i in range(motif_size-1)])
                gap_size_count[motif_size][gap_size] += 1
            else:
                for i in range(motif_size - 1):
                    gap_size = indices[i+1]-indices[i] -1
                    gap_size_count[motif_size][gap_size] += 1

        motif_sizes = list()
        gap_sizes = list()
        occurrence = list()
        for motif_size in gap_size_count:
            if sum(gap_size_count[motif_size].values()) != 0:
                for gap_size in gap_size_count[motif_size]:
                    motif_sizes.append(str(motif_size))
                    gap_sizes.append(gap_size)
                    occurrence.append(gap_size_count[motif_size][gap_size])

        gap_size_df = pd.DataFrame()
        gap_size_df["motif_size"] = motif_sizes
        gap_size_df["gap_size"] = gap_sizes
        gap_size_df["occurrence"] = occurrence

        return gap_size_df

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

    def _plot_gap_sizes(self, gap_size_df):
        file_path = self.result_path / f"gap_and_motif_size.html"

        if self.max_gap_size_only:
            title = "Maximum gap size in motif distribution"
            x_label = "Max gap size"
        else:
            title= "Distances between Amino Acids for all positions"
            x_label = "Gap size"

        gap_size_fig = px.bar(
            gap_size_df,
            x="gap_size",
            y="occurrence",
            color="motif_size",
            color_discrete_sequence=self._get_color_discrete_sequence(),
            template="plotly_white",
            title=title,
            labels={
                "gap_size": x_label,
                "occurrence": "Occurrence",
                "motif_size": "Motif size",
            },
        )

        gap_size_fig.update_layout(font={"size": 14}, xaxis={"tickmode": "linear"})
        gap_size_fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Gap size between amino acids in high-precision high-recall motifs",
        )

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

    def _plot(self, positional_aa_counts_df, gap_size_df) -> List[ReportOutput]:
        report_outputs = []
        report_outputs.append(self._plot_gap_sizes(gap_size_df))
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

    def _write_gap_size_table(self, gap_size_df) -> List[ReportOutput]:
        table_path = self.result_path / f"gap_size_table.csv"
        gap_size_df.to_csv(table_path, index=False, header=True)

        return ReportOutput(
            path=table_path,
            name=f"Gap size between amino acids in high-precision high-recall motifs for all motif sizes",
        )

    def check_prerequisites(self):
        valid_encodings = [MotifEncoder.__name__]

        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning(
                "PositonalMotifFrequencies: the dataset is not encoded, skipping this report..."
            )
            return False
        elif self.dataset.encoded_data.encoding not in valid_encodings:
            logging.warning(
                f"PositonalMotifFrequencies: the dataset encoding ({self.dataset.encoded_data.encoding}) was not in the list of valid "
                f"encodings ({valid_encodings}), skipping this report..."
            )
            return False
        elif self.max_gap_size_only is None:
            return False
        else:
            return True
