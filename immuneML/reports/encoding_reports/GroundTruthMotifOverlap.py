from pathlib import Path

import logging
import plotly.express as px
import numpy as np
import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper

from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class GroundTruthMotifOverlap(EncodingReport):
    """
    Creates report displaying overlap between learned motifs and groundtruth motifs

    # todo: arguments, yaml spec, explanation of format of highlight motifs file
    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1, groundtruth_motifs_path: str = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.groundtruth_motifs_path = groundtruth_motifs_path

    @classmethod
    def build_object(cls, **kwargs):
        location = GroundTruthMotifOverlap.__name__

        if "groundtruth_motifs_path" in kwargs and kwargs["groundtruth_motifs_path"] is not None:
            PositionalMotifHelper.check_motif_filepath(kwargs["groundtruth_motifs_path"], location, "groundtruth_motifs_path", expected_header="indices\tamino_acids\tn_sequences\n")

        return GroundTruthMotifOverlap(**kwargs)

    def _generate(self):
        PathBuilder.build(self.result_path)

        groundtruth_motifs, implant_rate_dict = self._read_groundtruth_motifs(self.groundtruth_motifs_path)

        learned_motifs = self.dataset.encoded_data.feature_names

        overlap_df = self._generate_overlap(learned_motifs, groundtruth_motifs, implant_rate_dict)
        output_table = self._write_output_table(overlap_df, self.result_path / "ground_truth_motif_overlap.tsv", name=None)
        output_figure = self._safe_plot(overlap_df=overlap_df)

        return ReportResult(
            name=self.name,
            output_figures=[output_figure] if output_figure is not None else [],
            output_tables=[output_table],
        )

    def _read_groundtruth_motifs(self, filepath):
        with open(filepath) as file:
            PositionalMotifHelper.check_file_header(file.readline(), filepath, expected_header="indices\tamino_acids\tn_sequences\n")
            groundtruth_motifs = []
            groundtruth_implant_rate = []
            for line in file.readlines():
                motif, implant_rate = self._get_motif_and_implant_rate(
                    line, motif_sep="\t"
                )
                groundtruth_motifs.append(motif)
                groundtruth_implant_rate.append(implant_rate)

        implant_rate_dict = {
            groundtruth_motifs[i]: groundtruth_implant_rate[i]
            for i in range(len(groundtruth_motifs))
        }
        return groundtruth_motifs, implant_rate_dict

    def _get_motif_and_implant_rate(self, string, motif_sep):
        indices_str, amino_acids_str, implant_rate = string.strip().split(motif_sep)
        motif = indices_str + "-" + amino_acids_str
        return motif, implant_rate

    def _generate_overlap(self, learned_motifs, groundtruth_motifs, implant_rate_dict):
        motif_size_list = list()
        implant_rate_list = list()
        max_overlap_list = list()

        for learned_motif in learned_motifs:
            motif_size = len(learned_motif.split("-")[0].replace("&", ""))
            for groundtruth_motif in groundtruth_motifs:
                max_overlap = self._get_max_overlap(learned_motif, groundtruth_motif)
                if max_overlap != 0:
                    motif_size_list.append(motif_size)
                    implant_rate_list.append(implant_rate_dict[groundtruth_motif])
                    max_overlap_list.append(max_overlap)

        df = pd.DataFrame()
        df["implant_rate"] = implant_rate_list
        df["max_overlap"] = max_overlap_list
        df["motif_size"] = motif_size_list

        return df

    def _get_max_overlap(self, learned_motif, groundtruth_motif):
        larger, smaller = groundtruth_motif, learned_motif
        if len(groundtruth_motif) < len(learned_motif):
            larger, smaller = learned_motif, groundtruth_motif

        larger_ind, larger_aa = PositionalMotifHelper.string_to_motif(
            larger, value_sep="&", motif_sep="-"
        )
        smaller_ind, smaller_aa = PositionalMotifHelper.string_to_motif(
            smaller, value_sep="&", motif_sep="-"
        )

        max_score = 0
        max_larger_index = len(larger_ind)
        max_smaller_index = len(smaller_ind)
        loop_counter = 0
        start = 0

        score = 0
        while loop_counter < max_larger_index:

            for larger_index, smaller_index in zip(
                range(start, max_larger_index), range(max_smaller_index)
            ):
                if (
                    larger_ind[larger_index] == smaller_ind[smaller_index]
                    and larger_aa[larger_index] == smaller_aa[smaller_index]
                ):
                    score += 1

            max_score = max(max_score, score)
            loop_counter += 1
            start = loop_counter

        return max_score

    def _get_color_discrete_sequence(self):
        return px.colors.qualitative.Pastel[:-1] + px.colors.qualitative.Set3

    def _plot(self, overlap_df) -> ReportOutput:
        file_path = self.result_path / f"motif_overlap.html"
        categories = np.sort([int(cat) for cat in overlap_df["implant_rate"].unique()])
        facet_barplot = px.histogram(
            overlap_df,
            x="implant_rate",
            labels={
                "implant_rate": "Implant rate of groundtruth motif",
                "max_groundtruth_overlap": "Max groundtruth overlap",
                "motif_size": "Motif size",
            },
            facet_col="max_overlap",
            color_discrete_sequence=self._get_color_discrete_sequence(),
            category_orders=dict(implant_rate=categories),
            facet_col_spacing=0.05,
            color="motif_size",
            title="Amount of overlapping motifs per implant rate",
            template="plotly_white"
        )
        facet_barplot.update_yaxes(matches=None, showticklabels=True)
        facet_barplot.update_layout(
            xaxis_title="Implant rate of groundtruth motif",
            yaxis_title="Total overlapping learned motifs",
        )
        facet_barplot.write_html(str(file_path))

        return ReportOutput(
            path=file_path, name="Amount of overlapping motifs per implant rate"
        )

    def check_prerequisites(self):
        valid_encodings = [MotifEncoder.__name__]

        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning(
                "GroundTruthMotifOverlap: the dataset is not encoded, skipping this report..."
            )
            return False
        elif self.dataset.encoded_data.encoding not in valid_encodings:
            logging.warning(
                f"GroundTruthMotifOverlap: the dataset encoding ({self.dataset.encoded_data.encoding}) was not in the list of valid "
                f"encodings ({valid_encodings}), skipping this report..."
            )
            return False
        else:
            return True