from pathlib import Path

import logging
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import (
    PositionalMotifHelper,
)
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.SequenceType import SequenceType
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class GroundtruthMotifOverlap(EncodingReport):
    """
    Creates report displaying overlap between learned motifs and groundtruth motifs
    """

    def __init__(
        self,
        dataset: Dataset = None,
        result_path: Path = None,
        name: str = None,
        number_of_processes: int = 1,
        highlight_motifs_path: str = None,
    ):
        super().__init__(
            dataset=dataset,
            result_path=result_path,
            name=name,
            number_of_processes=number_of_processes,
        )
        self.highlight_motifs_path = highlight_motifs_path

    @classmethod
    def build_object(cls, **kwargs):
        location = GroundtruthMotifOverlap.__name__

        if (
            "highlight_motifs_path" in kwargs
            and kwargs["highlight_motifs_path"] is not None
        ):
            PositionalMotifHelper.check_motif_filepath(
                kwargs["highlight_motifs_path"], location, "highlight_motifs_path"
            )

        return GroundtruthMotifOverlap(**kwargs)

    def _generate(self):
        PathBuilder.build(self.result_path)

        groundtruth_motifs, implant_rate_dict = self._read_highlight_motifs(
            self.highlight_motifs_path
        )

        learned_motifs = self.dataset.encoded_data.feature_names

        overlap_dataframe = self._generate_overlap(
            learned_motifs, groundtruth_motifs, implant_rate_dict
        )

        output_figures = self._plot(overlap_dataframe, implant_rate_dict)
        return ReportResult(
            name=self.name,
            output_figures=output_figures,
        )

    def _read_highlight_motifs(self, filepath):
        with open(filepath) as file:
            PositionalMotifHelper._check_file_header(file.readline(), filepath)
            groundtruth_motifs = []
            groundtruth_implant_rate = []
            for line in file.readlines():
                motif, implant_rate = self._get_motif_and_implant_rate(line, motif_sep="\t")
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

    def _generate_overlap(
        self, learned_motifs, groundtruth_motifs, implant_rate_dict
    ):
        overlap_df = pd.DataFrame({"learned motifs": learned_motifs})

        for gt_motif in groundtruth_motifs:
            overlap_df[gt_motif] = [
                self._get_max_overlap(l_motif, gt_motif)
                for l_motif in overlap_df["learned motifs"]
            ]

        overlap_df["max_groundtruth_overlap"] = overlap_df.drop(
            "learned motifs", axis=1
        ).max(axis=1)
        overlap_df["groundtruth_motif"] = overlap_df.drop(
            ["learned motifs", "max_groundtruth_overlap"], axis=1
        ).idxmax(axis=1)
        overlap_df["groundtruth_motif"] = overlap_df["groundtruth_motif"].apply(
            lambda x: int(implant_rate_dict[x])
        )
        overlap_df = overlap_df[["groundtruth_motif", "max_groundtruth_overlap"]]
        overlap_df = (
            overlap_df.max_groundtruth_overlap.groupby(
                [overlap_df.groundtruth_motif, overlap_df.max_groundtruth_overlap]
            )
            .sum()
            .unstack()
            .fillna(0)
            .astype(int)
        )
        overlap_df.drop(columns=overlap_df.iloc[:, 0], inplace=True)
        overlap_df.sort_index(axis=1, inplace=True, ascending=True)
        overlap_df.sort_index(axis=0, inplace=True, ascending=False)
        return overlap_df

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

        while loop_counter < max_larger_index:
            score = 0

            for larger_index, smaller_index in zip(range(start, max_larger_index), range(max_smaller_index)):
                if (
                    larger_ind[larger_index] == smaller_ind[smaller_index]
                    and larger_aa[larger_index] == smaller_aa[smaller_index]
                ):
                    score += 1

            max_score = max(max_score, score)
            loop_counter += 1
            start = loop_counter

        return max_score

    def _plot(self, overlap_dataframe, implant_rate_dict) -> ReportOutput:
        file_path = self.result_path / f"motif_overlap.html"
        heatmap = px.imshow(
            overlap_dataframe,
            text_auto=True,
            aspect="auto",
            x=[str(num) for num in overlap_dataframe.columns],
            y=[str(num) for num in overlap_dataframe.index],
        )
        heatmap.update_layout(
            xaxis_title="Maximum overlap with groundtruth motifs",
            yaxis_title="Implant rate of groundtruth motif",
        )
        heatmap.update_layout(
            coloraxis_colorbar=dict(
                title="Total overlap of learned motifs",
            ),
        )
        heatmap.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Overlap between learned and groundtruth motifs over implant rate",
        )

    def check_prerequisites(self):
        valid_encodings = [MotifEncoder.__name__]

        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning(
                "GroundtruthMotifOverlap: the dataset is not encoded, skipping this report..."
            )
            return False
        elif self.dataset.encoded_data.encoding not in valid_encodings:
            logging.warning(
                f"GroundtruthMotifOverlap: the dataset encoding ({self.dataset.encoded_data.encoding}) was not in the list of valid "
                f"encodings ({valid_encodings}), skipping this report..."
            )
            return False
        else:
            return True
