import copy
import dataclasses
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from bionumpy import DNAEncoding, AminoAcidEncoding
from bionumpy.bnpdataclass import BNPDataClass

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.GenModelAsTSV import GenModelAsTSV
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.signal_implanting.LigoImplanterState import LigoImplanterState
from immuneML.simulation.util.bnp_util import make_bnp_dataclass_object_from_dicts, merge_dataclass_objects
from immuneML.simulation.util.util import get_sequence_per_signal_count, make_sequences_from_gen_model, get_bnp_data, filter_out_illegal_sequences, \
    annotate_sequences, build_imgt_positions, choose_implant_position, check_iteration_progress, get_signal_sequence_count, get_custom_keys, \
    check_sequence_count, prepare_data_for_repertoire_obj, update_seqs_without_signal, update_seqs_with_signal, make_receptor_sequence_objects
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.PositionHelper import PositionHelper


class LigoImplanter:

    def __init__(self, state: LigoImplanterState):
        self._state = state

    MIN_RANGE_PROBABILITY = 1e-5

    @property
    def max_signals(self):
        return 0 if self._state.remove_seqs_with_signals else -1

    @property
    def fields(self) -> list:
        return [(field.name, field.type) for field in dataclasses.fields(GenModelAsTSV)] + \
               [(signal.id, int) for signal in self._state.all_signals] + \
               [(f"{signal.id}_positions", str) for signal in self._state.all_signals]

    @property
    def use_p_gens(self) -> bool:
        return (self._state.export_p_gens or self._state.keep_p_gen_dist) and self._state.sim_item.generative_model.can_compute_p_gens()

    def make_repertoires(self, path: Path) -> List[Repertoire]:

        self._gen_necessary_sequences(path)
        repertoires = self._make_repertoire_objects(path)
        return repertoires

    def _gen_necessary_sequences(self, path: Path):
        seqs_per_signal_count = get_sequence_per_signal_count(self._state.sim_item)
        self._make_sequence_paths(path)
        iteration = 0

        while sum(seqs_per_signal_count.values()) > 0 and iteration < self._state.max_iterations:
            sequences = self._make_background_sequences(path, iteration)

            if self._state.keep_p_gen_dist and iteration == 0:
                sequences = self._make_p_gen_histogram(sequences)

            if self._state.remove_seqs_with_signals:
                sequences = self._filter_background_sequences(sequences)

            sequences = self._implant_in_sequences(sequences, copy.deepcopy(seqs_per_signal_count))

            if self.use_p_gens:
                sequences = self._filter_using_p_gens(sequences)

            seqs_per_signal_count['no_signal'] = update_seqs_without_signal(seqs_per_signal_count['no_signal'], sequences,
                                                                            self._state.sequence_paths['no_signal'])
            seqs_per_signal_count = update_seqs_with_signal(copy.deepcopy(seqs_per_signal_count), sequences, self._state.all_signals,
                                                            self._state.sim_item.signals,
                                                            self._state.sequence_paths)

            check_iteration_progress(iteration, self._state.max_iterations)
            iteration += 1

        if iteration == self._state.max_iterations and sum(seqs_per_signal_count.values()) != 0:
            raise RuntimeError(f"{LigoImplanter.__name__}: maximum iterations were reached, but the simulation could not finish "
                               f"with parameters: {vars(self)}.\n")

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path) -> List[ReceptorSequence]:

        assert len(self._state.sim_item.signals) in [0, 1], f"RejectionSampler: for sequence datasets, only 0 or 1 signal per sequence are " \
                                                            f"supported, but {len(self._state.sim_item.signals)} were specified."

        self._gen_necessary_sequences(path)
        sequences = None

        if len(self._state.sim_item.signals) == 0:
            sequences = get_bnp_data(self._state.sequence_paths['no_signal'], self.fields)
        else:
            for signal in self._state.sim_item.signals:
                signal_sequences = get_bnp_data(self._state.sequence_paths[signal.id], self.fields)
                if sequences is not None:
                    sequences = merge_dataclass_objects([sequences, signal_sequences])
                else:
                    sequences = signal_sequences

        sequences = self._add_pgens(sequences)
        sequences = make_receptor_sequence_objects(sequences, self._state.all_signals, self._make_signal_metadata(), self._state.sim_item.immune_events)

        return sequences

    def _add_pgens(self, sequences: GenModelAsTSV) -> GenModelAsTSV:
        if sequences.p_gen is None and self.use_p_gens:
            sequences.p_gen = self._state.sim_item.generative_model.compute_p_gens(sequences, self._state.sequence_type)
        return sequences

    def _make_background_sequences(self, path, iteration: int) -> GenModelAsTSV:
        sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{iteration}_{self._state.sim_item.name}.tsv"
        make_sequences_from_gen_model(self._state.sim_item, self._state.sequence_batch_size, self._state.seed, sequence_path,
                                      self._state.sequence_type, False)
        return get_bnp_data(sequence_path)

    def _filter_background_sequences(self, sequences: GenModelAsTSV) -> GenModelAsTSV:
        annotated_sequences = annotate_sequences(sequences, self._state.sequence_type == SequenceType.AMINO_ACID, self._state.all_signals)
        if self._state.remove_seqs_with_signals:
            annotated_sequences = filter_out_illegal_sequences(annotated_sequences, self._state.sim_item, self._state.all_signals, self.max_signals)
        return annotated_sequences

    def _implant_in_sequences(self, sequences: GenModelAsTSV, seqs_per_signal_count: dict) -> GenModelAsTSV:

        sequence_lengths = getattr(sequences, self._state.sequence_type.value).lengths
        remaining_seq_mask = np.ones(len(sequences), dtype=bool)
        modified_sequence_dataclass_objs = []

        for signal in self._state.sim_item.signals:
            if seqs_per_signal_count[signal.id] > 0 and remaining_seq_mask.sum() > 0:
                modified_sequences = []
                motif_instances = self._make_motif_instances(signal, seqs_per_signal_count[signal.id])

                for instance in motif_instances:
                    suitable_seqs = np.argwhere(np.logical_and(remaining_seq_mask, sequence_lengths >= len(instance))).reshape(-1)
                    if suitable_seqs.shape[0] > 0:
                        sequence_index = np.random.choice(suitable_seqs, size=1)[0]
                        new_sequence = self._implant_in_sequence(sequences[sequence_index], signal, instance)
                        remaining_seq_mask[sequence_index] = False
                        modified_sequences.append(new_sequence)
                    else:
                        logging.warning(f"{LigoImplanter.__name__}: could not find a sequence to implant {instance} for signal {signal.id}, "
                                        f"skipping for now.")

                modified_sequences = make_bnp_dataclass_object_from_dicts(modified_sequences, signals=self._state.all_signals,
                                                                          base_class=GenModelAsTSV,
                                                                          field_type_map={"sequence": DNAEncoding, "sequence_aa": AminoAcidEncoding,
                                                                                          'p_gen': float})

                modified_sequence_dataclass_objs.append(modified_sequences)

                seqs_per_signal_count[signal.id] -= len(modified_sequences)

        if seqs_per_signal_count['no_signal'] > 0 and remaining_seq_mask.sum() > 0:
            sequences = self._add_info_to_no_signal_sequences(sequences[remaining_seq_mask])
            sequences = merge_dataclass_objects([sequences] + modified_sequence_dataclass_objs)
        else:
            sequences = merge_dataclass_objects(modified_sequence_dataclass_objs)

        return sequences

    def _add_info_to_no_signal_sequences(self, sequences: GenModelAsTSV) -> BNPDataClass:

        new_fields = {**{s.id: [0 for _ in range(len(sequences))] for s in self._state.all_signals},
                      **{f"{s.id}_positions": ["m" + "".join("0" for _ in range(getattr(sequences, self._state.sequence_type.value).lengths[i]))
                                               for i in range(len(sequences))]
                         for s in self._state.all_signals}}

        if self.use_p_gens and np.any(sequences.p_gen < 0):
            sequences.p_gen = self._state.sim_item.generative_model.compute_p_gens(sequences, self._state.sequence_type)

        return sequences.add_fields(new_fields)

    def _implant_in_sequence(self, sequence_row: GenModelAsTSV, signal: Signal, motif_instance: MotifInstance) -> dict:
        imgt_aa_positions = build_imgt_positions(len(getattr(sequence_row, self._state.sequence_type.value)), motif_instance,
                                                 sequence_row.region_type)

        limit = len(motif_instance)
        if self._state.sequence_type == SequenceType.NUCLEOTIDE:
            limit = limit * 3

        position_weights = PositionHelper.build_position_weights(signal.sequence_position_weights, imgt_aa_positions, limit)
        implant_position = choose_implant_position(imgt_aa_positions, position_weights)

        new_sequence = self._make_new_sequence(sequence_row, motif_instance, implant_position)

        if self.use_p_gens:
            new_sequence['p_gen'] = self._state.sim_item.generative_model.compute_p_gen(
                {key: val.to_string() if hasattr(val, "to_string") else val for key, val in new_sequence.items()},
                self._state.sequence_type)
        else:
            new_sequence['p_gen'] = -1.

        new_sequence[signal.id] = 1
        new_sequence[f'{signal.id}_positions'] = "m" + "".join("0" for _ in range(implant_position)) + "1" + \
                                                 "".join("0" for _ in
                                                         range(len(getattr(sequence_row, self._state.sequence_type.value)) - implant_position))

        zero_mask = "m" + "".join(["0" for _ in range(len(new_sequence[self._state.sequence_type.value]))])
        new_sequence = {**{f"{s.id}_positions": zero_mask for s in self._state.all_signals}, **new_sequence}

        return new_sequence

    def _make_new_sequence(self, sequence_row: GenModelAsTSV, motif_instance: MotifInstance, position) -> dict:
        if "/" in motif_instance.instance:
            motif_left, motif_right = motif_instance.instance.split("/")
        else:
            motif_left = motif_instance.instance
            motif_right = ""

        if self._state.sequence_type == SequenceType.NUCLEOTIDE:
            position *= 3
        sequence_string = getattr(sequence_row, self._state.sequence_type.value).to_string()

        gap_start = position + len(motif_left)
        gap_end = gap_start + motif_instance.gap

        new_sequence_string = sequence_string[:position] + motif_left + sequence_string[gap_start:gap_end] + motif_right + \
                              sequence_string[gap_end + len(motif_right):]

        sequence_dict = {'region_type': sequence_row.region_type, 'frame_type': '', 'v_call': sequence_row.v_call, 'j_call': sequence_row.j_call,
                         'sequence': '', 'sequence_aa': new_sequence_string}

        if self._state.sequence_type == SequenceType.NUCLEOTIDE:
            sequence_dict['sequence'] = new_sequence_string
            sequence_dict['sequence_aa'] = ReceptorSequence.nt_to_aa(new_sequence_string)

        return sequence_dict

    def _make_motif_instances(self, signal: Signal, seqs_count: int):
        if seqs_count > 0:
            instances = signal.make_motif_instances(seqs_count, self._state.sequence_type)

            if any(not isinstance(el, MotifInstance) for el in instances):
                raise NotImplementedError(
                    "When using implanting, V and J genes must not been set in the motifs -- V/J gene implanting is not supported.")

            return instances
        else:
            return []

    def _make_p_gen_histogram(self, sequences: GenModelAsTSV):
        self._check_if_can_compute_pgens()
        p_gens = self._state.sim_item.generative_model.compute_p_gens(sequences, self._state.sequence_type)
        sequences.p_gen = p_gens
        log_p_gens = np.log10(p_gens)
        hist, self._state.p_gen_bins = np.histogram(log_p_gens, bins=np.concatenate(
            ([np.NINF], np.histogram_bin_edges(log_p_gens, self._state.p_gen_bin_count), [np.PINF])),
                                                    density=False)
        self._state.target_p_gen_histogram = hist / len(sequences)

        zero_regions = self._state.target_p_gen_histogram == 0
        self._state.target_p_gen_histogram[zero_regions] = LigoImplanter.MIN_RANGE_PROBABILITY
        self._state.target_p_gen_histogram[np.logical_not(zero_regions)] -= \
            LigoImplanter.MIN_RANGE_PROBABILITY * np.sum(zero_regions) / np.sum(np.logical_not(zero_regions))

        return sequences

    def _filter_using_p_gens(self, sequences: GenModelAsTSV) -> Tuple[BNPDataClass, dict]:
        if np.any(sequences.p_gen == -1):
            missing_p_gens = sequences.p_gen == -1
            sequences.p_gen[missing_p_gens] = self._state.sim_item.generative_model.compute_p_gens(sequences[missing_p_gens],
                                                                                                   self._state.sequence_type)

        p_gens = np.log10(sequences.p_gen)
        sequence_bins = np.digitize(p_gens, self._state.p_gen_bins) - 1
        keep_sequences = np.random.uniform(0, 1, len(sequences)) <= self._state.target_p_gen_histogram[sequence_bins]

        return sequences[keep_sequences]

    def _make_repertoire_objects(self, path: Path):
        repertoires = []
        used_seq_count = {**{'no_signal': 0}, **{signal.id: 0 for signal in self._state.sim_item.signals}}
        repertoires_path = PathBuilder.build(path / "repertoires")

        for i in range(self._state.sim_item.number_of_examples):
            seqs_no_signal_count = self._state.sim_item.receptors_in_repertoire_count \
                                   - get_signal_sequence_count(repertoire_count=1, sim_item=self._state.sim_item) * len(self._state.sim_item.signals)

            custom_columns = get_custom_keys(self._state.all_signals, [('p_gen', float)])

            sequences, used_seq_count = self._get_no_signal_sequences(used_seq_count=used_seq_count, seqs_no_signal_count=seqs_no_signal_count,
                                                                      columns=custom_columns)
            sequences, used_seq_count = self._add_signal_sequences(sequences, custom_columns, used_seq_count)

            check_sequence_count(self._state.sim_item, sequences)

            repertoire = self._make_repertoire_from_sequences(sequences, repertoires_path)

            repertoires.append(repertoire)

        shutil.rmtree(path / "tmp", ignore_errors=True)

        return repertoires

    def _make_repertoire_from_sequences(self, sequences: BNPDataClass, repertoires_path) -> Repertoire:
        metadata = {**self._make_signal_metadata(), **self._state.sim_item.immune_events}
        rep_data = prepare_data_for_repertoire_obj(self._state.all_signals, sequences, [('p_gen', float)])
        return Repertoire.build(**rep_data, path=repertoires_path, metadata=metadata)

    def _make_signal_metadata(self) -> dict:
        return {**{signal.id: True if not self._state.sim_item.is_noise else False for signal in self._state.sim_item.signals},
                **{signal.id: False for signal in self._state.all_signals if signal not in self._state.sim_item.signals}}

    def _add_signal_sequences(self, sequences, columns, used_seq_count: dict):

        for signal in self._state.sim_item.signals:

            skip_rows = used_seq_count[signal.id]
            n_rows = round(self._state.sim_item.receptors_in_repertoire_count * self._state.sim_item.repertoire_implanting_rate)

            sequences_sig = get_bnp_data(self._state.sequence_paths[signal.id], columns)[skip_rows:skip_rows + n_rows]

            used_seq_count[signal.id] += n_rows

            if sequences is None:
                sequences = sequences_sig
            else:
                sequences = merge_dataclass_objects([sequences, sequences_sig])

        return sequences, used_seq_count

    def _get_no_signal_sequences(self, used_seq_count: dict, seqs_no_signal_count: int, columns):
        if self._state.sequence_paths['no_signal'].is_file() and seqs_no_signal_count > 0:
            skip_rows = used_seq_count['no_signal']
            used_seq_count['no_signal'] = used_seq_count['no_signal'] + seqs_no_signal_count
            return get_bnp_data(self._state.sequence_paths['no_signal'], columns)[skip_rows:skip_rows + seqs_no_signal_count], used_seq_count
        else:
            return None, used_seq_count

    def _make_sequence_paths(self, path: Path):
        tmp_path = PathBuilder.build(path / 'implanted_sequences')
        self._state.sequence_paths = {signal.id: tmp_path / f'{signal.id}.tsv' for signal in self._state.sim_item.signals}
        self._state.sequence_paths['no_signal'] = tmp_path / 'no_signal.tsv'

    def _check_if_can_compute_pgens(self):
        if not self._state.sim_item.generative_model.can_compute_p_gens():
            raise RuntimeError(f"{LigoImplanter.__name__}: generative model of class {type(self._state.sim_item.generative_model).__name__} cannot "
                               f"compute sequence generation probabilities. Use other generative model or set keep_p_gen_dist parameter to False.")
