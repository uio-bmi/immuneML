from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.Signal import Signal


@dataclass
class RejectionSampler:
    simulation_item: LIgOSimulationItem
    sequence_type: SequenceType
    all_signals: List[Signal]
    seed: int = 1

    MAX_SIGNALS_PER_SEQUENCE = 1
    MIN_SEQUENCES_TO_GENERATE = 50

    def make_repertoire_sequences(self, path: Path):
        sequence_count = self.simulation_item.number_of_receptors_in_repertoire
        sequence_with_signal_count = {signal.id: int(sequence_count * self.simulation_item.repertoire_implanting_rate)
                                      for signal in self.simulation_item.signals}
        sequence_without_signal_count = sequence_count - sum(sequence_with_signal_count.values())
        all_signal_ids = [signal.id for signal in self.all_signals]

        sequences = pd.DataFrame(index=np.arange(sequence_count), columns=self.simulation_item.generative_model.OUTPUT_COLUMNS)

        while sum(sequence_with_signal_count.values()) != 0 or sequence_without_signal_count != 0:
            background_sequences = self.simulation_item.generative_model.generate_sequences(
                max(self.simulation_item.number_of_receptors_in_repertoire, RejectionSampler.MIN_SEQUENCES_TO_GENERATE), seed=self.seed,
                path=path / f"tmp_{self.seed}.tsv", sequence_type=self.sequence_type)
            self.seed += 1

            signal_matrix = self.get_signal_matrix(background_sequences)
            background_sequences, signal_matrix = self.filter_out_illegal_sequences(background_sequences, signal_matrix)

            if sequence_without_signal_count > 0:
                seqs = background_sequences[signal_matrix.sum(axis=1) == 0][:sequence_without_signal_count]
                index_start = sequence_count - sequence_without_signal_count - sum(sequence_with_signal_count.values())
                sequences.iloc[index_start: index_start + seqs.shape[0]] = seqs
                sequence_without_signal_count -= len(seqs)

            for signal in self.simulation_item.signals:
                if sequence_with_signal_count[signal.id] > 0:
                    selection = signal_matrix[signal.id]
                    seqs = background_sequences.loc[selection][:sequence_with_signal_count[signal.id]]
                    index_start = sequence_count - sequence_without_signal_count - sum(sequence_with_signal_count.values())
                    sequences.iloc[index_start: index_start + seqs.shape[0]] = seqs
                    sequence_with_signal_count[signal.id] -= len(seqs)

        return sequences

    def filter_out_illegal_sequences(self, sequences: pd.DataFrame, signal_matrix: np.ndarray):
        if RejectionSampler.MAX_SIGNALS_PER_SEQUENCE != 1:
            raise NotImplementedError

        sim_signal_ids = [signal.id for signal in self.simulation_item.signals]
        other_signals = [signal.id not in sim_signal_ids for signal in self.all_signals]
        background_to_keep = np.logical_and(signal_matrix.sum(axis=1) <= RejectionSampler.MAX_SIGNALS_PER_SEQUENCE,
                                            signal_matrix[:, other_signals] == 0 if any(other_signals) else 1)
        sequences = sequences[background_to_keep]
        signal_matrix = signal_matrix[background_to_keep]

        return sequences, signal_matrix

    def get_signal_matrix(self, sequences: pd.DataFrame) -> np.ndarray:
        sequence_mask = sequences.apply(lambda row: self.contains_signals(row.to_dict()), axis=1, result_type='expand')
        sequence_mask.columns = [signal.id for signal in self.all_signals]
        return sequence_mask

    def contains_signals(self, sequence: dict):
        return [signal.is_in(sequence, self.sequence_type) for signal in self.all_signals]

    def make_receptors(self):
        raise NotImplementedError