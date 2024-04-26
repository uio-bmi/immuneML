from pathlib import Path

import logging
import plotly.express as px
import pandas as pd
from typing import List

from immuneML.data_model.dataset import SequenceDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import (
    PositionalMotifHelper,
)
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class PositionalMotifFrequencies(EncodingReport):
    """
    This report must be used in combination with the :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`.
    Plots a stacked bar plot of amino acid occurrence at different indices in any given dataset, along with a plot
    investigating motif continuity which displays a bar plot of the gap sizes between the amino acids in the motifs in
    the given dataset. Note that a distance of 1 means that the amino acids are continuous (next to each other).

    **Specification arguments:**

    - motif_color_map (dict): Optional mapping between motif lengths and specific colors to be used. Example:

        motif_color_map:
            1: #66C5CC
            2: #F6CF71
            3: #F89C74


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_pos_motif_report:
                    PositionalMotifFrequencies:
                        motif_color_map:

    """

    @classmethod
    def build_object(cls, **kwargs):
        if "motif_color_map" in kwargs:
            ParameterValidator.assert_type_and_value(kwargs["motif_color_map"], dict, PositionalMotifFrequencies.__name__, "motif_color_map")
            kwargs["motif_color_map"] = {str(key): value for key, value in kwargs["motif_color_map"].items()}

        return PositionalMotifFrequencies(**kwargs)

    def __init__(self, dataset: SequenceDataset = None, result_path: Path = None,
                 motif_color_map: dict = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.motif_color_map = motif_color_map

    def get_sequence_length(self):
        my_sequence = next(self.dataset.get_data())
        return len(my_sequence.get_sequence())

    def _generate(self):
        PathBuilder.build(self.result_path)

        motifs = self.dataset.encoded_data.feature_names
        positional_aa_counts_df = self._get_positional_aa_counts(motifs)
        max_gap_size_df = self._get_max_gap_sizes(motifs)
        total_gap_size_df = self._get_total_gap_sizes(motifs)


        positional_aa_counts_table = self._write_output_table(positional_aa_counts_df,
                                                              file_path=self.result_path / f"positional_aa_counts.csv",
                                                              name="Frequencies of amino acids found in the high-precision high-recall motifs")

        max_gap_size_table = self._write_output_table(max_gap_size_df,
                                                  file_path=self.result_path / f"max_gap_size_table.csv",
                                                  name="Maximum gap sizes within motifs")

        total_gap_size_table = self._write_output_table(total_gap_size_df,
                                                  file_path=self.result_path / f"total_gap_size_table.csv",
                                                  name="Total (summed) gap sizes within motifs")

        output_figure1 = self._safe_plot(positional_aa_counts_df=positional_aa_counts_df, plot_callable="_plot_positional_aa_counts")
        output_figure2 = self._safe_plot(gap_size_df=max_gap_size_df,  x="max_gap_size", plot_callable="_plot_gap_sizes")
        output_figure3 = self._safe_plot(gap_size_df=total_gap_size_df,  x="total_gap_size", plot_callable="_plot_gap_sizes")

        return ReportResult(
            name=self.name,
            output_figures=[fig for fig in [output_figure1, output_figure2, output_figure3] if fig is not None],
            output_tables=[max_gap_size_table, total_gap_size_table, positional_aa_counts_table],
        )

    def _get_total_gap_sizes(self, motifs):
        data = {"motif_size": [],
              "total_gap_size": [],
              "occurrence": []}

        gap_size_count = {}

        for motif in motifs:
            motif_indices, amino_acids = PositionalMotifHelper.string_to_motif(motif, "&", "-")
            total_gap_size = (max(motif_indices) - min(motif_indices)) - len(motif_indices) + 1
            motif_size = len(motif_indices)

            if motif_size not in gap_size_count:
                gap_size_count[motif_size] = {total_gap_size: 1}
            else:
                if total_gap_size not in gap_size_count[motif_size]:
                    gap_size_count[motif_size][total_gap_size] = 1
                else:
                    gap_size_count[motif_size][total_gap_size] += 1

        for motif_size, counts in gap_size_count.items():
            for total_gap_size, occurrence in counts.items():
                data["motif_size"].append(motif_size)
                data["total_gap_size"].append(total_gap_size)
                data["occurrence"].append(occurrence)

        return pd.DataFrame(data)

    def _get_max_gap_sizes(self, motifs):
        gap_size_count = {
            motif_size: {
                gap_size: 0 for gap_size in range(0, self.get_sequence_length() - 1)
            }
            for motif_size in range(self.get_sequence_length())
        }

        for motif in motifs:
            indices, amino_acids = PositionalMotifHelper.string_to_motif(motif, "&", "-")
            motif_size = len(indices)

            if motif_size > 1:
                gap_size = max([indices[i+1]-indices[i] -1 for i in range(motif_size-1)])
                gap_size_count[motif_size][gap_size] += 1

        motif_sizes = list()
        max_gap_sizes = list()
        occurrence = list()
        for motif_size in gap_size_count:
            if sum(gap_size_count[motif_size].values()) != 0:
                for gap_size in gap_size_count[motif_size]:
                    motif_sizes.append(str(motif_size))
                    max_gap_sizes.append(gap_size)
                    occurrence.append(gap_size_count[motif_size][gap_size])

        gap_size_df = pd.DataFrame()
        gap_size_df["motif_size"] = motif_sizes
        gap_size_df["max_gap_size"] = max_gap_sizes
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

        # start counting positions at 1
        positional_aa_counts_df.index = [idx+1 for idx in list(positional_aa_counts_df.index)]

        return positional_aa_counts_df

    def _plot_gap_sizes(self, gap_size_df, x):
        file_path = self.result_path / f"{x}.html"

        gap_size_df["occurrence_total"] = gap_size_df.groupby("motif_size")["occurrence"].transform(sum)
        gap_size_df["occurrence_percentage"] = gap_size_df["occurrence"] / gap_size_df["occurrence_total"]

        x_label = x.replace("_", " ").capitalize()

        if self.motif_color_map is not None:
            color_discrete_map = self.motif_color_map
            color_discrete_sequence = None
        else:
            color_discrete_map = None
            color_discrete_sequence = self._get_color_discrete_sequence()

        gap_size_df = gap_size_df.sort_values(by=x)
        gap_size_df["motif_size"] = gap_size_df["motif_size"].astype(str)

        gap_size_fig = px.line(
            gap_size_df,
            x=x,
            y="occurrence_percentage",
            color="motif_size",
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=dict(motif_size=sorted([int(rate) for rate in gap_size_df["motif_size"].unique()])),
            template="plotly_white",
            markers=True,
            labels={
                x: x_label,
                "occurrence_percentage": "Percentage of motifs",
                "motif_size": "Motif size",
            },
        )
        gap_size_fig.layout.yaxis.tickformat = ',.0%'

        gap_size_fig.update_layout(font={"size": 14}, xaxis={"tickmode": "linear"})
        gap_size_fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Gap size between amino acids in high-precision high-recall motifs",
        )

    def _plot_positional_aa_counts(self, positional_aa_counts_df):
        file_path = self.result_path / f"positional_motif_frequencies.html"

        # reverse sort column names makes amino acids stack alphabetically in bar chart
        positional_aa_counts_df = positional_aa_counts_df[sorted(positional_aa_counts_df.columns)[::-1]]

        positional_aa_counts_fig = px.bar(
            positional_aa_counts_df,
            labels={
                "index": "Amino acid position",
                "value": "Frequency across motifs",
            },
            text="variable",
            color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
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

        return True
