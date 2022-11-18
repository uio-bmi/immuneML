import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.bnp_util import merge_dataclass_objects, add_field_to_bnp_dataclass
from immuneML.simulation.util.util import get_signal_sequence_count, get_sequence_per_signal_count, make_sequences_from_gen_model, get_bnp_data, \
    annotate_sequences, filter_out_illegal_sequences, write_bnp_data
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
    export_pgens: bool = None

    MAX_SIGNALS_PER_SEQUENCE = 1
    MAX_MOTIF_POSITION_LENGTH = 10

    @property
    def is_amino_acid(self):
        return self.sequence_type == SequenceType.AMINO_ACID

    def make_repertoires(self, path: Path) -> List[Repertoire]:
        repertoires_path = PathBuilder.build(path / "repertoires")

        sequence_per_signal_count = get_sequence_per_signal_count(self.sim_item)

        self._make_background_sequences(path / f"tmp_{self.sim_item.name}", sequence_per_signal_count)
        repertoires = []

        for i in range(self.sim_item.number_of_examples):
            seqs_no_signal_count = self.sim_item.receptors_in_repertoire_count - get_signal_sequence_count(repertoire_count=1,
                                                                                                           sim_item=self.sim_item) * len(
                self.sim_item.signals)

            custom_columns = self._get_custom_keys(with_p_gens=False)

            sequences = self._get_no_signal_sequences(repertoire_index=i, seqs_no_signal_count=seqs_no_signal_count, columns=custom_columns)
            sequences = self._add_signal_sequences(sequences, custom_columns, repertoire_index=i)
            sequences = self._add_pgens(sequences)

            self._check_sequence_count(sequences)

            repertoire = self._make_repertoire_from_sequences(sequences, repertoires_path)

            repertoires.append(repertoire)

        shutil.rmtree(path / "tmp", ignore_errors=True)

        return repertoires

    def _get_custom_keys(self, with_p_gens: bool = True):
        keys = [(sig.id, int) for sig in self.all_signals] + [(f'{signal.id}_positions', str) for signal in self.all_signals]
        if with_p_gens:
            keys += [('p_gen', float)]
        return keys

    def _make_repertoire_from_sequences(self, sequences, repertoires_path) -> Repertoire:
        metadata = {**self._make_signal_metadata(), **self.sim_item.immune_events}
        custom_keys = self._get_custom_keys()

        repertoire = Repertoire.build(**{**{key: val for key, val in sequences.to_dict('list').items() if key not in custom_keys},
                                         **{"custom_lists": sequences[custom_keys].to_dict('list')}},
                                      path=repertoires_path, metadata=metadata)
        return repertoire

    def _make_signal_metadata(self) -> dict:
        return {**{signal.id: True if not self.sim_item.is_noise else False for signal in self.sim_item.signals},
                **{signal.id: False for signal in self.all_signals if signal not in self.sim_item.signals}}

    def _get_no_signal_sequences(self, repertoire_index: int, seqs_no_signal_count, columns):
        if self.seqs_no_signal_path.is_file() and seqs_no_signal_count > 0:
            skip_rows = repertoire_index * seqs_no_signal_count + 1
            return get_bnp_data(self.seqs_no_signal_path, columns)[skip_rows:skip_rows+seqs_no_signal_count]
        else:
            return None

    def _add_signal_sequences(self, sequences, columns, repertoire_index: int):

        for signal in self.sim_item.signals:

            skip_rows = get_signal_sequence_count(repertoire_count=repertoire_index, sim_item=self.sim_item) + 1
            n_rows = int(self.sim_item.receptors_in_repertoire_count * self.sim_item.repertoire_implanting_rate)

            sequences_sig = get_bnp_data(self.seqs_with_signal_path[signal.id], columns)[skip_rows:skip_rows+n_rows]

            if sequences is None:
                sequences = sequences_sig
            else:
                sequences = merge_dataclass_objects([sequences, sequences_sig])

        return sequences

    def _check_sequence_count(self, sequences: pd.DataFrame):
        assert sequences.shape[0] == self.sim_item.receptors_in_repertoire_count, \
            f"{RejectionSampler.__name__}: error when simulating repertoire, needed {self.sim_item.receptors_in_repertoire_count} sequences, " \
            f"but got {sequences.shape[0]}."

    def _make_background_sequences(self, path: Path, sequence_per_signal_count: dict):
        PathBuilder.build(path)
        self._setup_tmp_sequence_paths(path)
        iteration = 1

        while (sum(sequence_per_signal_count.values()) != 0) and iteration <= self.max_iterations:
            sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{self.seed}_{self.sim_item.name}.tsv"

            needs_seqs_with_signal = (sum(sequence_per_signal_count.values()) - sequence_per_signal_count['no_signal']) > 0
            make_sequences_from_gen_model(self.sim_item, self.sequence_batch_size, self.seed, sequence_path, self.sequence_type,
                                          needs_seqs_with_signal)
            self.seed += 1

            background_sequences = get_bnp_data(sequence_path)
            annotated_sequences = annotate_sequences(background_sequences, self.is_amino_acid, self.all_signals)
            annotated_sequences = filter_out_illegal_sequences(annotated_sequences, self.sim_item, self.all_signals,
                                                               RejectionSampler.MAX_SIGNALS_PER_SEQUENCE)

            sequence_per_signal_count['no_signal'] = self._update_seqs_without_signal(sequence_per_signal_count['no_signal'], annotated_sequences)
            sequence_per_signal_count = self._update_seqs_with_signal(sequence_per_signal_count, annotated_sequences)

            if iteration == int(self.max_iterations * 0.75):
                logging.warning(f"Iteration {iteration} out of {self.max_iterations} max iterations reached during rejection sampling.")
            iteration += 1

        if iteration == self.max_iterations and sum(sequence_per_signal_count.values()) != 0:
            raise RuntimeError(f"{RejectionSampler.__name__}: maximum iterations were reached, but the simulation could not finish "
                               f"with parameters: {vars(self)}.\n")

    def _setup_tmp_sequence_paths(self, path):
        if self.seqs_with_signal_path is None:
            self.seqs_with_signal_path = {signal.id: path / f"sequences_with_signal_{signal.id}.tsv" for signal in self.sim_item.signals}
        if self.seqs_no_signal_path is None:
            self.seqs_no_signal_path = path / "sequences_no_signal.tsv"

    def _update_seqs_without_signal(self, max_count, annotated_sequences):
        if max_count > 0:
            selection = annotated_sequences.get_signal_matrix().sum(axis=1) == 0
            data_to_write = annotated_sequences[selection][:max_count]
            if len(data_to_write) > 0:
                write_bnp_data(data=data_to_write, path=self.seqs_no_signal_path)
            return max_count - len(data_to_write)
        else:
            return max_count

    def _update_seqs_with_signal(self, max_counts: dict, annotated_sequences):

        all_signal_ids = [signal.id for signal in self.all_signals]
        signal_matrix = annotated_sequences.get_signal_matrix()

        for signal in self.sim_item.signals:
            if max_counts[signal.id] > 0:
                selection = signal_matrix[:, all_signal_ids.index(signal.id)].astype(bool)
                data_to_write = annotated_sequences[selection][:max_counts[signal.id]]
                if len(data_to_write) > 0:
                    write_bnp_data(data=data_to_write, path=self.seqs_with_signal_path[signal.id])
                max_counts[signal.id] -= len(data_to_write)

        return max_counts

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path) -> List[ReceptorSequence]:

        assert len(self.sim_item.signals) in [0, 1], f"RejectionSampler: for sequence datasets, only 0 or 1 signal per sequence are supported, " \
                                                     f"but {len(self.sim_item.signals)} were specified."

        PathBuilder.build(path)
        sequences = None

        seqs_per_signal_count = get_sequence_per_signal_count(self.sim_item)
        self._make_background_sequences(path / f"tmp_{self.sim_item.name}", seqs_per_signal_count)

        metadata = {**{signal.id: False if self.sim_item.is_noise else True for signal in self.sim_item.signals},
                    **{signal.id: False for signal in self.all_signals if signal not in self.sim_item.signals}}

        if len(self.sim_item.signals) == 0:
            sequences = pd.read_csv(self.seqs_no_signal_path, sep='\t')
        else:
            for signal in self.sim_item.signals:
                if sequences is not None:
                    sequences = pd.concat([sequences, pd.read_csv(self.seqs_with_signal_path[signal.id], sep='\t')], ignore_index=True, axis=0)
                else:
                    sequences = pd.read_csv(self.seqs_with_signal_path[signal.id], sep='\t')

        sequences = self._add_pgens(sequences)
        custom_params_keys = self._get_custom_keys()

        sequences = [ReceptorSequence(seq['sequence_aa'], seq['sequence'], identifier=uuid.uuid4().hex,
                                      metadata=SequenceMetadata(
                                          custom_params={**metadata, **{key: str(seq[key]) if 'position' in key else getattr(seq, key, None)
                                                                        for key in custom_params_keys}, **self.sim_item.immune_events},
                                          v_call=seq['v_call'] if 'v_call' in seq else None,
                                          j_call=seq['j_call'] if 'j_call' in seq else None,
                                          region_type=seq['region_type'])) for _, seq in
                     sequences.iterrows()]

        shutil.rmtree(path / "tmp", ignore_errors=True)

        return sequences

    def _add_pgens(self, sequences):
        if not hasattr(sequences, 'p_gen'):
            if self.export_pgens and self.sim_item.generative_model.can_compute_p_gens():
                p_gens = self.sim_item.generative_model.compute_p_gens(sequences, self.sequence_type)
            else:
                p_gens = None

            sequences = add_field_to_bnp_dataclass(sequences, 'p_gen', float, p_gens)

        return sequences
