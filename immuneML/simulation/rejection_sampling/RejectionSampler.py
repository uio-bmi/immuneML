import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class RejectionSampler:
    sim_item: LIgOSimulationItem
    sequence_type: SequenceType
    all_signals: List[Signal]
    sequence_batch_size: int
    max_iterations: int
    seqs_no_signal_path: Path = None
    seqs_with_signal_path: dict = None
    seed: int = 1

    MAX_SIGNALS_PER_SEQUENCE = 1

    def make_repertoires(self, path: Path) -> List[Repertoire]:
        repertoires_path = PathBuilder.build(path / "repertoires")

        sequence_count = self.sim_item.number_of_receptors_in_repertoire * self.sim_item.number_of_examples
        sequence_with_signal_count = {signal.id: self._get_signal_sequence_count(repertoire_count=self.sim_item.number_of_examples)
                                      for signal in self.sim_item.signals}
        sequence_without_signal_count = sequence_count - sum(sequence_with_signal_count.values())

        self._generate_sequences(path / "tmp", sequence_without_signal_count, sequence_with_signal_count)
        repertoires = []

        for i in range(self.sim_item.number_of_examples):

            seqs_no_signal_count = self.sim_item.number_of_receptors_in_repertoire - self._get_signal_sequence_count(repertoire_count=1) * len(self.sim_item.signals)

            column_names = self.sim_item.generative_model.OUTPUT_COLUMNS + [s.id for s in self.all_signals]

            sequences = self._get_no_signal_sequences(repertoire_index=i, seqs_no_signal_count=seqs_no_signal_count, column_names=column_names)
            sequences = self._add_signal_sequences(sequences, column_names, repertoire_index=i)

            self._check_sequence_count(sequences)

            repertoire = self._make_repertoire_from_sequences(sequences, repertoires_path)

            repertoires.append(repertoire)

        shutil.rmtree(path / "tmp")
        return repertoires

    def _make_repertoire_from_sequences(self, sequences, repertoires_path) -> Repertoire:
        metadata = self._make_signal_metadata()

        custom_params = sequences[[signal.id for signal in self.all_signals]]

        repertoire = Repertoire.build(**{**{key: val for key, val in sequences.to_dict('list').items()
                                            if key not in [s.id for s in self.all_signals]}, **{"custom_lists": custom_params.to_dict('list')}},
                                      path=repertoires_path, metadata=metadata)
        return repertoire

    def _get_signal_sequence_count(self, repertoire_count: int) -> int:
        return round(self.sim_item.number_of_receptors_in_repertoire * self.sim_item.repertoire_implanting_rate) * repertoire_count

    def _make_signal_metadata(self) -> dict:
        return {**{signal.id: True if not self.sim_item.is_noise else False for signal in self.sim_item.signals},
                **{signal.id: False for signal in self.all_signals if signal not in self.sim_item.signals}}

    def _get_no_signal_sequences(self, repertoire_index: int, seqs_no_signal_count, column_names):
        if self.seqs_no_signal_path.is_file() and seqs_no_signal_count > 0:
            skip_rows = repertoire_index * seqs_no_signal_count + 1
            return pd.read_csv(self.seqs_no_signal_path, skiprows=skip_rows, nrows=seqs_no_signal_count, names=column_names)
        else:
            return None

    def _add_signal_sequences(self, sequences: pd.DataFrame, column_names, repertoire_index: int):
        for signal in self.sim_item.signals:
            skip_rows = self._get_signal_sequence_count(repertoire_count=repertoire_index) + 1
            tmp_df = pd.read_csv(self.seqs_with_signal_path[signal.id], names=column_names, skiprows=skip_rows,
                                 nrows=int(self.sim_item.number_of_receptors_in_repertoire * self.sim_item.repertoire_implanting_rate))
            if sequences is None:
                sequences = tmp_df
            else:
                sequences = pd.concat([sequences, tmp_df], axis=0, ignore_index=True)

        return sequences

    def _check_sequence_count(self, sequences: pd.DataFrame):
        assert sequences.shape[0] == self.sim_item.number_of_receptors_in_repertoire, \
            f"{RejectionSampler.__name__}: error when simulating repertoire, needed {self.sim_item.number_of_receptors_in_repertoire} sequences, " \
            f"but got {sequences.shape[0]}."

    def _generate_sequences(self, path: Path, sequence_without_signal_count: int, sequence_with_signal_count: dict):
        PathBuilder.build(path)
        self._setup_tmp_sequence_paths(path)
        iteration = 1

        while (sum(sequence_with_signal_count.values()) != 0 or sequence_without_signal_count != 0) and iteration <= self.max_iterations:
            background_sequences = self.sim_item.generative_model.generate_sequences(
                max(self.sim_item.number_of_receptors_in_repertoire, self.sequence_batch_size), seed=self.seed,
                path=path / f"gen_model/tmp_{self.seed}.tsv", sequence_type=self.sequence_type)
            self.seed += 1

            signal_matrix = self.get_signal_matrix(background_sequences)
            background_sequences, signal_matrix = self.filter_out_illegal_sequences(background_sequences, signal_matrix)

            sequence_without_signal_count = self._update_seqs_without_signal(sequence_without_signal_count, signal_matrix, background_sequences)
            sequence_with_signal_count = self._update_seqs_with_signal(sequence_with_signal_count, signal_matrix, background_sequences)

            if iteration == int(self.max_iterations * 0.75):
                logging.warning(f"Iteration {iteration} out of {self.max_iterations} max iterations reached.")
            iteration += 1

    def _setup_tmp_sequence_paths(self, path):
        if self.seqs_with_signal_path is None:
            self.seqs_with_signal_path = {signal.id: path / f"sequences_with_signal_{signal.id}.csv" for signal in self.sim_item.signals}
        if self.seqs_no_signal_path is None:
            self.seqs_no_signal_path = path / "sequences_no_signal.csv"

    def _update_seqs_without_signal(self, sequence_without_signal_count, signal_matrix, background_sequences: pd.DataFrame):
        if sequence_without_signal_count > 0:
            seqs = background_sequences[signal_matrix.sum(axis=1) == 0][:sequence_without_signal_count]
            self._store_sequences(seqs, self.seqs_no_signal_path)
            return sequence_without_signal_count - len(seqs)
        else:
            return sequence_without_signal_count

    def _update_seqs_with_signal(self, sequence_with_signal_count: dict, signal_matrix, background_sequences: pd.DataFrame):
        all_signal_ids = [signal.id for signal in self.all_signals]

        for signal in self.sim_item.signals:
            if sequence_with_signal_count[signal.id] > 0:
                selection = signal_matrix[signal.id]
                seqs = background_sequences.loc[selection][:sequence_with_signal_count[signal.id]]
                seqs[all_signal_ids] = pd.DataFrame([[True if id == signal.id else False for id in all_signal_ids]], index=seqs.index)
                self._store_sequences(seqs, self.seqs_with_signal_path[signal.id])
                sequence_with_signal_count[signal.id] -= len(seqs)

        return sequence_with_signal_count

    def _store_sequences(self, seqs: pd.DataFrame, path: Path):
        if path.is_file():
            seqs.to_csv(str(path), mode='a', header=False, index=False)
        else:
            seqs.to_csv(str(path), mode='w', header=True, index=False)

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path) -> List[ReceptorSequence]:

        assert len(self.sim_item.signals) in [0, 1], f"RejectionSampler: for sequence datasets, only 0 or 1 signal per sequence are supported, " \
                                                     f"but {len(self.sim_item.signals)} were specified."

        PathBuilder.build(path)

        seqs_no_signal_count = 0 if len(self.sim_item.signals) == 0 else self.sim_item.number_of_examples
        sequence_with_signal_count = {signal.id: self.sim_item.number_of_examples for signal in self.sim_item.signals}

        self._generate_sequences(path / "tmp", seqs_no_signal_count, sequence_with_signal_count)

        sequences = []

        metadata = {**{signal.id: False if self.sim_item.is_noise else True for signal in self.sim_item.signals},
                    **{signal.id: False for signal in self.all_signals if signal not in self.sim_item.signals}}

        if seqs_no_signal_count > 0:
            sequences = pd.read_csv(self.seqs_no_signal_path)
        else:
            for signal in self.sim_item.signals:
                sequences = pd.read_csv(self.seqs_with_signal_path[signal.id])

        sequences = [ReceptorSequence(seq['sequence_aa'], seq['sequence'], identifier=uuid.uuid4().hex,
                                      metadata=SequenceMetadata(custom_params=metadata, v_call=seq['v_call'] if 'v_call' in seq else None,
                                                                j_call=seq['j_call'] if 'j_call' in seq else None)) for seq in sequences.iterrows()]

        shutil.rmtree(path / "tmp")
        return sequences

    def filter_out_illegal_sequences(self, sequences: pd.DataFrame, signal_matrix: np.ndarray):
        if RejectionSampler.MAX_SIGNALS_PER_SEQUENCE != 1:
            raise NotImplementedError

        sim_signal_ids = [signal.id for signal in self.sim_item.signals]
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
