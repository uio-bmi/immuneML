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

        definitions:
            datasets:
                my_data:
                    ...
            reports:
                my_sld_report: SequenceLengthDistribution

        instructions:
                instruction_1:
                    type: ExploratoryAnalysis
                    analyses:
                        my_sld_analysis:
                            dataset: unpaired_data
                            report: my_sld_report
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
        file_path = self.plot(normalized_sequence_lengths)
        return ReportResult(type(self).__name__, output_figures=[ReportOutput(file_path, "sequence length distribution plot")])

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

    def plot(self, normalized_sequence_length):

        plt.style.use('ggplot')

        x = OrderedDict(sorted(normalized_sequence_length.items(), key=lambda item: item[0]))

        plt.bar(list(x.keys()), list(x.values()), alpha=0.45, color="b")
        plt.xticks(list(x.keys()), list(x.keys()))
        plt.grid(True, color='k', alpha=0.07, axis='y')
        plt.xlabel("Lengths")
        plt.ylabel("Frequency")
        plt.title("Sequence length distribution")

        PathBuilder.build(self.result_path)

        file_path = self.result_path + "sequence_length_distribution.png"
        plt.savefig(file_path, transparent=True)
        return file_path

