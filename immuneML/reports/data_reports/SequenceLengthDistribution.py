import warnings
from collections import Counter
from pathlib import Path
from typing import Union

import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SequenceLengthDistribution(DataReport):
    """
    Generates a histogram of the lengths of the sequences in a repertoire or sequence dataset.

    **Specification arguments:**

    - sequence_type (str): whether to check the length of amino acid or nucleotide sequences; default value is 'amino_acid'

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_sld_report:
                    SequenceLengthDistribution:
                        sequence_type: amino_acid

    """

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_sequence_type(kwargs)

        return SequenceLengthDistribution(**{**kwargs, 'sequence_type': SequenceType[kwargs['sequence_type'].upper()]})

    def __init__(self, dataset: Union[RepertoireDataset, SequenceDataset] = None, batch_size: int = 1, result_path: Path = None, number_of_processes: int = 1,
                 sequence_type: SequenceType = SequenceType.AMINO_ACID, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.batch_size = batch_size
        self.sequence_type = sequence_type

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset) or isinstance(self.dataset, SequenceDataset):
            return True
        else:
            warnings.warn("SequenceLengthDistribution: report can be generated only from repertoire and sequence datasets. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:
        sequence_lengths = self._get_sequence_lengths()
        PathBuilder.build(self.result_path)
        df = pd.DataFrame({"counts": list(sequence_lengths.values()), 'sequence_lengths': list(sequence_lengths.keys())})
        df.to_csv(self.result_path / 'sequence_length_distribution.csv', index=False)

        report_output_fig = self._safe_plot(df=df, output_written=False)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(name=self.name,
                            info="A histogram of the lengths of the sequences in a dataset.",
                            output_figures=output_figures, output_tables=[ReportOutput(self.result_path / 'sequence_length_distribution.csv',
                                                                                       'lengths of sequences in the dataset')])

    def _get_sequence_lengths(self) -> Counter:

        if isinstance(self.dataset, RepertoireDataset):
            sequence_lengths = Counter()

            for repertoire in self.dataset.get_data(self.batch_size):
                seq_lengths = self._count_in_repertoire(repertoire)
                sequence_lengths += seq_lengths
        else:
            sequence_lengths = {}
            for sequence in self.dataset.get_data():
                l = len(sequence.get_sequence(self.sequence_type))
                if l in sequence_lengths:
                    sequence_lengths[l] += 1
                else:
                    sequence_lengths[l] = 1

        return sequence_lengths

    def _count_in_repertoire(self, repertoire: Repertoire) -> Counter:
        c = Counter([len(sequence.get_sequence(self.sequence_type)) for sequence in repertoire.sequences])
        return c

    def _plot(self, df: pd.DataFrame) -> ReportOutput:

        figure = px.bar(df, x="sequence_lengths", y="counts")
        figure.update_layout(xaxis=dict(tickmode='array', tickvals=df["sequence_lengths"]), yaxis=dict(tickmode='array', tickvals=df["counts"]),
                             template="plotly_white")
        figure.update_traces(marker_color=px.colors.diverging.Tealrose[0])
        PathBuilder.build(self.result_path)

        file_path = self.result_path / "sequence_length_distribution.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="Sequence length distribution plot")

