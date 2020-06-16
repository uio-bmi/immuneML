import warnings
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.data_reports.DataReport import DataReport
from source.util.PathBuilder import PathBuilder


class SequenceLengthDistribution(DataReport):
    """
    Generates a histogram of the lengths of the sequences in a RepertoireDataset.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sld_report: SequenceLengthDistribution

    """

    @classmethod
    def build_object(cls, **kwargs):
        return SequenceLengthDistribution(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, batch_size: int = 1, result_path: str = None, name: str = None):
        DataReport.__init__(self, dataset=dataset, result_path=result_path, name=name)
        self.batch_size = batch_size

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn("SequenceLengthDistribution: report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def generate(self) -> ReportResult:
        normalized_sequence_lengths = self.get_normalized_sequence_lengths()
        report_output_fig = self._safe_plot(normalized_sequence_lengths=normalized_sequence_lengths, output_written=False)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(type(self).__name__, output_figures=output_figures)

    def get_normalized_sequence_lengths(self) -> Counter:
        sequence_lenghts = Counter()

        for repertoire in self.dataset.get_data(self.batch_size):
            seq_lengths = self.count_in_repertoire(repertoire)
            sequence_lenghts += seq_lengths

        total = sum(sequence_lenghts.values())

        for key in sequence_lenghts:
            sequence_lenghts[key] /= total

        return sequence_lenghts

    def count_in_repertoire(self, repertoire: Repertoire) -> Counter:
        c = Counter([len(sequence.get_sequence()) for sequence in repertoire.sequences])
        return c

    def _plot(self, normalized_sequence_lengths):

        figure, ax = plt.subplots()
        plt.style.use('ggplot')

        x = OrderedDict(sorted(normalized_sequence_lengths.items(), key=lambda item: item[0]))

        plt.bar(list(x.keys()), list(x.values()), alpha=0.45, color="b")
        plt.xticks(list(x.keys()), list(x.keys()))
        plt.grid(True, color='k', alpha=0.07, axis='y')
        plt.xlabel("Lengths")
        plt.ylabel("Frequency")
        plt.title("Sequence length distribution")
        plt.box(on=None)

        PathBuilder.build(self.result_path)

        file_path = self.result_path + "sequence_length_distribution.png"
        figure.savefig(file_path, transparent=True)
        return ReportOutput(path=file_path, name="sequence length distribution plot")

