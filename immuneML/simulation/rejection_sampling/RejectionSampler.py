from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.Signal import Signal


@dataclass
class RejectionSampler:
    simulation_item: LIgOSimulationItem
    sequence_type: SequenceType
    all_signals: List[Signal]

    MAX_SIGNALS_PER_SEQUENCE = 1

    def make_repertoire(self, path: Path):
        sequence_count = self.simulation_item.number_of_receptors_in_repertoire
        sequence_with_signal_count = {signal.id: int(sequence_count * self.simulation_item.repertoire_implanting_rate)
                                      for signal in self.simulation_item.signals}
        sequence_without_signal_count = sequence_count - sum(sequence_with_signal_count.values())
        all_signal_ids = [signal.id for signal in self.all_signals]

        sequences = []

        while sum(sequence_with_signal_count.values()) != 0 and sequence_without_signal_count != 0:
            background_sequences = self.simulation_item.generative_model.generate_sequences(self.simulation_item.number_of_receptors_in_repertoire,
                                                                                            seed=1, path=path, sequence_type=self.sequence_type)

            signal_matrix = self.get_signal_matrix(background_sequences)
            background_sequences, signal_matrix = self.filter_out_illegal_sequences(background_sequences, signal_matrix)

            if sequence_without_signal_count > 0:
                seqs = background_sequences[signal_matrix.sum(axis=1) == 0].tolist()[:sequence_without_signal_count]
                sequences.append(seqs)
                sequence_without_signal_count -= len(seqs)

            for signal in self.simulation_item.signals:
                if sequence_with_signal_count[signal.id] > 0:
                    seqs = background_sequences[signal_matrix[:, all_signal_ids.index(signal.id)] == 1].tolist()[:sequence_with_signal_count[signal.id]]
                    sequences.append(seqs)
                    sequence_with_signal_count[signal.id] -= len(seqs)

        return sequences

    def filter_out_illegal_sequences(self, sequences, signal_matrix):
        if RejectionSampler.MAX_SIGNALS_PER_SEQUENCE != 1:
            raise NotImplementedError

        sim_signal_ids = [signal.id for signal in self.simulation_item.signals]
        other_signals = [signal.id not in sim_signal_ids for signal in self.all_signals]
        background_to_keep = np.logical_and(signal_matrix.sum(axis=1) < RejectionSampler.MAX_SIGNALS_PER_SEQUENCE, signal_matrix[:, other_signals] == 0)
        sequences = sequences[background_to_keep]
        signal_matrix = signal_matrix[background_to_keep]

        return sequences, signal_matrix

    def get_signal_matrix(self, sequences: np.ndarray) -> np.ndarray:
        sequence_mask = np.ndarray([self.contains_signals(sequence) for sequence in sequences])
        return sequence_mask

    def contains_signals(self, sequence):
        return [signal.is_in(sequence, self.sequence_type) for signal in self.all_signals]

    def make_receptors(self):
        raise NotImplementedError
