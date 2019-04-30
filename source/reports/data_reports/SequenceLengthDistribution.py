from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

from source.data_model.dataset.Dataset import Dataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.reports.data_reports.DataReport import DataReport
from source.util.PathBuilder import PathBuilder


class SequenceLengthDistribution(DataReport):

    def generate(self, dataset: Dataset, result_path: str, params: dict):
        normalized_sequence_lengths = self.get_normalized_sequence_lengths(dataset, params["batch_size"])
        self.plot(normalized_sequence_lengths, result_path)

    def get_normalized_sequence_lengths(self, dataset: Dataset, batch_size: int) -> Counter:
        sequence_lenghts = Counter()

        for repertoire in dataset.get_data(batch_size):
            seq_lengths = self.count_in_repertoire(repertoire)
            sequence_lenghts += seq_lengths

        total = sum(sequence_lenghts.values())

        for key in sequence_lenghts:
            sequence_lenghts[key] /= total

        return sequence_lenghts

    def count_in_repertoire(self, repertoire: Repertoire) -> Counter:
        c = Counter([len(sequence.get_sequence()) for sequence in repertoire.sequences])
        return c

    def plot(self, normalized_sequence_length, result_path):

        plt.style.use('ggplot')

        x = OrderedDict(sorted(normalized_sequence_length.items(), key=lambda item: item[0]))

        plt.bar(x.keys(), x.values(), alpha=0.45, color="b")
        plt.xticks(list(x.keys()), list(x.keys()))
        plt.grid(True, color='k', alpha=0.07, axis='y')
        plt.xlabel("Lengths")
        plt.ylabel("Frequency")
        plt.title("Sequence length distribution")

        PathBuilder.build(result_path)

        plt.savefig(result_path + "sequence_length_distribution.png", transparent=True)

