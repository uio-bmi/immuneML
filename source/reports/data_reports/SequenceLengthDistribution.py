import warnings
from collections import Counter

import pandas as pd
import plotly.express as px
from pathlib import Path

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.data_reports.DataReport import DataReport
from source.util.PathBuilder import PathBuilder


class SequenceLengthDistribution(DataReport):
    """
    Generates a histogram of the lengths of the sequences in a RepertoireDataset.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sld_report: SequenceLengthDistribution

    """

    @classmethod
    def build_object(cls, **kwargs):
        return SequenceLengthDistribution(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, batch_size: int = 1, result_path: Path = None, name: str = None):
        DataReport.__init__(self, dataset=dataset, result_path=result_path, name=name)
        self.batch_size = batch_size

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn("SequenceLengthDistribution: report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def generate(self) -> ReportResult:
        sequence_lengths = self.get_sequence_lengths()
        report_output_fig = self._plot(sequence_lengths=sequence_lengths)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(type(self).__name__, output_figures=output_figures)

    def get_sequence_lengths(self) -> Counter:
        sequence_lenghts = Counter()

        for repertoire in self.dataset.get_data(self.batch_size):
            seq_lengths = self.count_in_repertoire(repertoire)
            sequence_lenghts += seq_lengths

        return sequence_lenghts

    def count_in_repertoire(self, repertoire: Repertoire) -> Counter:
        c = Counter([len(sequence.get_sequence()) for sequence in repertoire.sequences])
        return c

    def _plot(self, sequence_lengths: Counter):

        df = pd.DataFrame({"counts": list(sequence_lengths.values()), 'sequence_lengths': list(sequence_lengths.keys())})

        figure = px.bar(df, x="sequence_lengths", y="counts")
        figure.update_layout(xaxis=dict(tickmode='array', tickvals=df["sequence_lengths"]), yaxis=dict(tickmode='array', tickvals=df["counts"]),
                             title="Sequence length distribution", template="plotly_white")
        figure.update_traces(marker_color=px.colors.diverging.Tealrose[0])
        PathBuilder.build(self.result_path)

        file_path = self.result_path / "sequence_length_distribution.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="sequence length distribution plot")

