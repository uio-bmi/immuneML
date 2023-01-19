from pathlib import Path

import logging
import plotly.express as px
import pandas as pd
from typing import List

from immuneML.data_model.dataset import SequenceDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class MotifOverlap(EncodingReport):
    """
    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None,
            number_of_processes: int = 1, highlight_motifs_path: str = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.highlight_motifs_path = highlight_motifs_path

    @classmethod
    def build_object(cls, **kwargs):

        if "highlight_motifs_path" in kwargs and kwargs["highlight_motifs_path"] is not None:
            PositionalMotifHelper.check_motif_filepath(kwargs["highlight_motifs_path"], location, "highlight_motifs_path")

        return MotifOverlap(**kwargs)

    def _generate(self):
        PathBuilder.build(self.result_path)

        encoded_dataset = encode(self.dataset)
        return ReportResult(
            name=self.name,
            output_figures=output_figures,
            output_tables=gap_size_tables + [positional_aa_counts_table],
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

    def _get_label_config(self):
        return LabelConfiguration([self._get_label()])

    def _get_label(self):
        label_name = self._get_label_name()
        label_values = list(set(self.dataset.encoded_data.labels[label_name]))
        positive_class = self._get_positive_class()

       return Label(name=label_name, values=label_values, positive_class=positive_class)

    def _get_label_name(self):
        return list(self.dataset.encoded_data.labels.keys())[0]

    def _get_positive_class(self):
        return self.dataset.encoded_data.info["positive_class"]

    def _get_encoder(self):
        encoder = MotifEncoder(label=self._get_label_name(),
                               name=f"motif_encoder_{self.name}")

        encoder.learned_motif_filepath = self.dataset.encoded_data.info["learned_motif_filepath"]

        return encoder

    def _encode(self, dataset):
        encoder = self._get_encoder()
        params = EncoderParams(result_path=self.result_path / "encoded_dataset",
                               label_config=self._get_label_config(),
                               pool_size=self.number_of_processes,
                               learn_model=False)

        return encoder.encode(dataset, params)

    def _plot(self, positional_aa_counts_df, gap_size_dict) -> List[ReportOutput]:
        report_outputs = self._plot_gap_sizes(gap_size_dict)
        report_outputs.append(self._plot_positional_aa_counts(positional_aa_counts_df))

        return report_outputs if len(report_outputs) != 0 else None

    def check_prerequisites(self):
        valid_encodings = [MotifEncoder.__name__]

        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning("MotifOverlap: the dataset is not encoded, skipping this report...")
            return False
        elif self.dataset.encoded_data.encoding not in valid_encodings:
            logging.warning(f"MotifOverlap: the dataset encoding ({self.dataset.encoded_data.encoding}) was not in the list of valid "
                            f"encodings ({valid_encodings}), skipping this report...")
            return False
        else:
            return True
