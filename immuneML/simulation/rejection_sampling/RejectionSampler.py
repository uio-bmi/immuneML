import logging
import shutil
import uuid
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import bionumpy as bnp
import numpy as np
import pandas as pd
from bionumpy.encodings import BaseEncoding
from bionumpy.string_matcher import RegexMatcher, StringMatcher

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.generative_models.GenModelAsTSV import GenModelAsTSV
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
    export_pgens: bool = None

    MAX_SIGNALS_PER_SEQUENCE = 1
    MAX_MOTIF_POSITION_LENGTH = 10

    @property
    def is_amino_acid(self):
        return self.sequence_type == SequenceType.AMINO_ACID

    def make_repertoires(self, path: Path) -> List[Repertoire]:
        repertoires_path = PathBuilder.build(path / "repertoires")

        sequence_count = self.sim_item.number_of_receptors_in_repertoire * self.sim_item.number_of_examples
        sequence_with_signal_count = {signal.id: self._get_signal_sequence_count(repertoire_count=self.sim_item.number_of_examples)
                                      for signal in self.sim_item.signals}
        sequence_without_signal_count = sequence_count - sum(sequence_with_signal_count.values())

        self._make_background_sequences(path / "tmp", sequence_without_signal_count, sequence_with_signal_count)
        repertoires = []

        for i in range(self.sim_item.number_of_examples):
            seqs_no_signal_count = self.sim_item.number_of_receptors_in_repertoire - self._get_signal_sequence_count(repertoire_count=1) * len(
                self.sim_item.signals)

            column_names = self.sim_item.generative_model.OUTPUT_COLUMNS + self._get_custom_keys(with_p_gens=False)

            sequences = self._get_no_signal_sequences(repertoire_index=i, seqs_no_signal_count=seqs_no_signal_count, column_names=column_names)
            sequences = self._add_signal_sequences(sequences, column_names, repertoire_index=i)
            sequences = self._add_pgens(sequences)

            self._check_sequence_count(sequences)

            repertoire = self._make_repertoire_from_sequences(sequences, repertoires_path)

            repertoires.append(repertoire)

        shutil.rmtree(path / "tmp", ignore_errors=True)
        return repertoires

    def _get_custom_keys(self, with_p_gens: bool = True):
        keys = [sig.id for sig in self.all_signals] + [f'{signal.id}_positions' for signal in self.all_signals]
        if with_p_gens:
            keys += ['p_gen']
        return keys

    def _make_repertoire_from_sequences(self, sequences, repertoires_path) -> Repertoire:
        metadata = self._make_signal_metadata()
        custom_keys = self._get_custom_keys()

        repertoire = Repertoire.build(**{**{key: val for key, val in sequences.to_dict('list').items() if key not in custom_keys},
                                         **{"custom_lists": sequences[custom_keys].to_dict('list')}},
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
            return pd.read_csv(self.seqs_no_signal_path, skiprows=skip_rows, nrows=seqs_no_signal_count, names=column_names, sep='\t',
                               dtype={col: str for col in column_names if 'position' in col})
        else:
            return None

    def _add_signal_sequences(self, sequences: pd.DataFrame, column_names, repertoire_index: int):
        for signal in self.sim_item.signals:
            skip_rows = self._get_signal_sequence_count(repertoire_count=repertoire_index) + 1
            tmp_df = pd.read_csv(self.seqs_with_signal_path[signal.id], names=column_names, skiprows=skip_rows, sep='\t',
                                 nrows=int(self.sim_item.number_of_receptors_in_repertoire * self.sim_item.repertoire_implanting_rate),
                                 dtype={col: str for col in column_names if 'position' in col})
            if sequences is None:
                sequences = tmp_df
            else:
                sequences = pd.concat([sequences, tmp_df], axis=0, ignore_index=True)

        return sequences

    def _check_sequence_count(self, sequences: pd.DataFrame):
        assert sequences.shape[0] == self.sim_item.number_of_receptors_in_repertoire, \
            f"{RejectionSampler.__name__}: error when simulating repertoire, needed {self.sim_item.number_of_receptors_in_repertoire} sequences, " \
            f"but got {sequences.shape[0]}."

    def _make_sequences_from_generative_model(self, sequence_path: Path, needs_seqs_with_signal: bool):
        self.sim_item.generative_model.generate_sequences(self.sequence_batch_size, seed=self.seed, path=sequence_path,
                                                          sequence_type=self.sequence_type)

        if self.sim_item.generative_model.can_generate_from_skewed_gene_models() and needs_seqs_with_signal:
            v_genes = sorted(
                list(set(chain.from_iterable([[motif.v_call for motif in signal.motifs if motif.v_call] for signal in self.sim_item.signals]))))
            j_genes = sorted(
                list(set(chain.from_iterable([[motif.j_call for motif in signal.motifs if motif.j_call] for signal in self.sim_item.signals]))))

            self.sim_item.generative_model.generate_from_skewed_gene_models(v_genes=v_genes, j_genes=j_genes, seed=self.seed, path=sequence_path,
                                                                            sequence_type=self.sequence_type, batch_size=self.sequence_batch_size)

    def _make_background_sequences(self, path: Path, sequence_without_signal_count: int, sequence_with_signal_count: dict):
        PathBuilder.build(path)
        self._setup_tmp_sequence_paths(path)
        iteration = 1

        while (sum(sequence_with_signal_count.values()) != 0 or sequence_without_signal_count != 0) and iteration <= self.max_iterations:
            sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{self.seed}.tsv"

            self._make_sequences_from_generative_model(sequence_path, needs_seqs_with_signal=sum(sequence_with_signal_count.values()) != 0)
            self.seed += 1

            # TODO: close file here after bnp.open (check in new bionumpy version)
            background_sequences = bnp.open(sequence_path, mode='full',
                                            buffer_type=bnp.delimited_buffers.get_bufferclass_for_datatype(GenModelAsTSV, delimiter="\t",
                                                                                                           has_header=True))
            signal_matrix, signal_positions = self.get_signal_matrix(background_sequences)
            legal_indices = self.filter_out_illegal_sequences(signal_matrix)
            signal_matrix = signal_matrix[legal_indices]
            signal_positions = signal_positions.iloc[legal_indices]
            background_sequences = background_sequences[legal_indices]

            sequence_without_signal_count = self._update_seqs_without_signal(sequence_without_signal_count, signal_matrix, background_sequences)
            sequence_with_signal_count = self._update_seqs_with_signal(sequence_with_signal_count, signal_matrix, background_sequences,
                                                                       signal_positions)

            if iteration == int(self.max_iterations * 0.75):
                logging.warning(f"Iteration {iteration} out of {self.max_iterations} max iterations reached during rejection sampling.")
            iteration += 1

    def _setup_tmp_sequence_paths(self, path):
        if self.seqs_with_signal_path is None:
            self.seqs_with_signal_path = {signal.id: path / f"sequences_with_signal_{signal.id}.tsv" for signal in self.sim_item.signals}
        if self.seqs_no_signal_path is None:
            self.seqs_no_signal_path = path / "sequences_no_signal.tsv"

    def _update_seqs_without_signal(self, sequence_without_signal_count, signal_matrix, background_sequences: pd.DataFrame):
        if sequence_without_signal_count > 0:
            all_signal_ids = [s.id for s in self.all_signals]
            seqs = pd.DataFrame(
                {key: getattr(background_sequences, key)[signal_matrix.sum(axis=1) == 0][:sequence_without_signal_count].to_sequences()
                 for key in self.sim_item.generative_model.OUTPUT_COLUMNS})
            seqs = self._init_signal_positions(seqs, all_signal_ids)
            seqs[all_signal_ids] = pd.DataFrame([[False for _ in all_signal_ids]], index=seqs.index)
            self._store_sequences(seqs, self.seqs_no_signal_path)
            return sequence_without_signal_count - len(seqs)
        else:
            return sequence_without_signal_count

    def _update_seqs_with_signal(self, sequence_with_signal_count: dict, signal_matrix, background_sequences: pd.DataFrame, signal_positions):
        all_signal_ids = [signal.id for signal in self.all_signals]

        for signal in self.sim_item.signals:
            if sequence_with_signal_count[signal.id] > 0:
                selection = signal_matrix[:, all_signal_ids.index(signal.id)]
                if np.any(selection):
                    seqs = pd.DataFrame({key: getattr(background_sequences, key)[selection][:sequence_with_signal_count[signal.id]].to_sequences()
                                         for key in self.sim_item.generative_model.OUTPUT_COLUMNS})
                    seqs[all_signal_ids] = pd.DataFrame([[True if id == signal.id else False for id in all_signal_ids]], index=seqs.index)
                    seqs = self._init_signal_positions(seqs, all_signal_ids)
                    seqs.update(pd.DataFrame({f'{signal.id}_positions': signal_positions.loc[selection][f'{signal.id}_positions'].values}))
                    self._store_sequences(seqs, self.seqs_with_signal_path[signal.id])
                    sequence_with_signal_count[signal.id] -= len(seqs)

        return sequence_with_signal_count

    def _init_signal_positions(self, seqs, all_signal_ids):
        zero_positions = 'm' + seqs['sequence_aa' if self.is_amino_acid else 'sequence'].str.replace("([A-Z])", "0").astype(str)
        for signal_id in all_signal_ids:
            seqs[f'{signal_id}_positions'] = zero_positions

        return seqs

    def _store_sequences(self, seqs: pd.DataFrame, path: Path):
        if path.is_file():
            seqs.to_csv(str(path), mode='a', header=False, index=False, sep='\t')
        else:
            seqs.to_csv(str(path), mode='w', header=True, index=False, sep='\t')

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path) -> List[ReceptorSequence]:

        assert len(self.sim_item.signals) in [0, 1], f"RejectionSampler: for sequence datasets, only 0 or 1 signal per sequence are supported, " \
                                                     f"but {len(self.sim_item.signals)} were specified."

        PathBuilder.build(path)
        sequences = None

        seqs_no_signal_count = 0 if len(self.sim_item.signals) > 0 else self.sim_item.number_of_examples
        sequence_with_signal_count = {signal.id: self.sim_item.number_of_examples // len(self.sim_item.signals) for signal in self.sim_item.signals}

        self._make_background_sequences(path / "tmp", seqs_no_signal_count, sequence_with_signal_count)

        metadata = {**{signal.id: False if self.sim_item.is_noise else True for signal in self.sim_item.signals},
                    **{signal.id: False for signal in self.all_signals if signal not in self.sim_item.signals}}

        if seqs_no_signal_count > 0:
            sequences = pd.read_csv(self.seqs_no_signal_path, sep='\t')

        for signal in self.sim_item.signals:
            if sequences is not None:
                sequences = pd.concat([sequences, pd.read_csv(self.seqs_with_signal_path[signal.id], sep='\t')], ignore_index=True, axis=0)
            else:
                sequences = pd.read_csv(self.seqs_with_signal_path[signal.id], sep='\t')

        sequences = self._add_pgens(sequences)
        custom_params_keys = self._get_custom_keys()

        sequences = [ReceptorSequence(seq['sequence_aa'], seq['sequence'], identifier=uuid.uuid4().hex,
                                      metadata=SequenceMetadata(custom_params={**metadata, **{key: str(seq[key]) if 'position' in key else getattr(seq, key, None)
                                                                                              for key in custom_params_keys}},
                                                                v_call=seq['v_call'] if 'v_call' in seq else None,
                                                                j_call=seq['j_call'] if 'j_call' in seq else None,
                                                                region_type=seq['region_type'])) for _, seq in
                     sequences.iterrows()]

        shutil.rmtree(path / "tmp", ignore_errors=True)

        return sequences

    def _add_pgens(self, sequences: pd.DataFrame) -> pd.DataFrame:
        if 'p_gen' not in sequences.columns:
            if self.export_pgens and self.sim_item.generative_model.can_compute_p_gens():
                sequences['p_gen'] = self.sim_item.generative_model.compute_p_gens(sequences, self.sequence_type)
            else:
                sequences['p_gen'] = None

        return sequences

    def filter_out_illegal_sequences(self, signal_matrix: np.ndarray):
        if RejectionSampler.MAX_SIGNALS_PER_SEQUENCE != 1:
            raise NotImplementedError

        sim_signal_ids = [signal.id for signal in self.sim_item.signals]
        other_signals = [signal.id not in sim_signal_ids for signal in self.all_signals]
        background_to_keep = np.logical_and(signal_matrix.sum(axis=1) <= RejectionSampler.MAX_SIGNALS_PER_SEQUENCE,
                                            np.array(signal_matrix[:, other_signals] == 0 if any(other_signals) else 1).all(axis=1))

        return background_to_keep

    def get_signal_matrix(self, sequences) -> Tuple[np.ndarray, pd.DataFrame]:

        encoding = bnp.encodings.AminoAcidEncoding if self.is_amino_acid else bnp.encodings.ACTGEncoding

        sequence_array = bnp.as_sequence_array(sequences.sequence_aa if self.is_amino_acid else sequences.sequence, encoding=encoding)
        v_call_array = bnp.as_sequence_array(sequences.v_call, encoding=bnp.encodings.BaseEncoding)
        j_call_array = bnp.as_sequence_array(sequences.j_call, encoding=bnp.encodings.BaseEncoding)

        signal_matrix = np.zeros((len(sequence_array), len(self.all_signals)))

        signal_positions = pd.DataFrame("", index=np.arange(len(sequence_array)), dtype=str,
                                        columns=[f'{signal.id}_positions' for signal in self.all_signals])

        for index, signal in enumerate(self.all_signals):
            signal_pos_col = None
            for motifs, v_call, j_call in signal.get_all_motif_instances(self.sequence_type):
                matches_gene = self._match_genes(v_call, v_call_array, j_call, j_call_array)
                matches = None

                for motif in motifs:

                    matches_motif = self._match_motif(motif, encoding, sequence_array)
                    if matches is None:
                        matches = np.logical_and(matches_motif, matches_gene)
                    else:
                        matches = np.logical_or(matches, np.logical_and((matches_motif, matches_gene)))

                signal_pos_col = np.logical_or(signal_pos_col, matches) if signal_pos_col is not None else matches
                signal_matrix[:, index] = np.logical_or(signal_matrix[:, index], np.logical_or.reduce(matches, axis=1))

            np_mask = np.where(signal_pos_col.ravel(), "1", "0")
            signal_positions.iloc[:, index] = ['m' + "".join(np_mask[start:end]) for start, end in
                                               zip(signal_pos_col.shape.starts, signal_pos_col.shape.ends)]

        return signal_matrix.astype(dtype=np.bool), signal_positions

    def _match_genes(self, v_call, v_call_array, j_call, j_call_array):

        if v_call is not None and v_call != "":
            matcher = StringMatcher(v_call, encoding=BaseEncoding)
            matches_gene = np.any(matcher.rolling_window(v_call_array), axis=1)
        else:
            matches_gene = np.ones(len(v_call_array))

        if j_call is not None and j_call != "":
            matcher = StringMatcher(j_call, encoding=BaseEncoding)
            matches_gene = np.logical_and(matches_gene, np.any(matcher.rolling_window(j_call_array), axis=1))

        return matches_gene.reshape(-1, 1).astype(bool)

    def _match_motif(self, motif, encoding, sequence_array):
        matcher = RegexMatcher(motif, encoding=encoding)
        matches = matcher.rolling_window(sequence_array)
        return matches
