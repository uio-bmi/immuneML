import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from bionumpy.bnpdataclass import bnpdataclass

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.bnp_util import make_bnp_dataclass_from_dicts, add_fields_to_bnp_dataclass, merge_dataclass_objects
from immuneML.simulation.util.util import get_sequence_per_signal_count, make_sequences_from_gen_model, get_bnp_data, filter_out_illegal_sequences, \
    annotate_sequences, build_imgt_positions, choose_implant_position, add_field_to_bnp_dataclass
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.PositionHelper import PositionHelper


@dataclass
class LigoImplanter:
    sim_item: LIgOSimulationItem = None
    sequence_type: SequenceType = None
    all_signals: List[Signal] = None
    sequence_batch_size: int = None
    seed: int = None
    export_p_gens: bool = None
    keep_p_gen_dist: bool = None
    remove_seqs_with_signals: bool = None
    max_iterations: int = None
    p_gen_threshold: float = None
    target_p_gen_histogram = None
    current_p_gen_histogram = None
    p_gen_bin_count: int = None
    keep_low_p_gen_proba: float = None
    sequence_paths: dict = None

    @property
    def max_signals(self):
        return 0 if self.remove_seqs_with_signals else -1

    def make_repertoires(self, path: Path) -> List[Repertoire]:

        seqs_per_signal_count = get_sequence_per_signal_count(self.sim_item)
        self._make_sequence_paths(path)
        iteration = 0

        while sum(seqs_per_signal_count.values()) > 0 and iteration < self.max_iterations:
            sequences = self._make_background_sequences(path)

            if self.keep_p_gen_dist and iteration == 0:
                self._make_p_gen_histogram(sequences)

            if self.remove_seqs_with_signals:
                sequences = self._filter_background_sequences(sequences)

            sequences, seqs_per_signal_count = self._implant_in_sequences(sequences, seqs_per_signal_count)

            if self.keep_p_gen_dist or self.p_gen_threshold:
                seqs_per_signal_count = self._filter_using_p_gens(sequences, seqs_per_signal_count)

            if iteration == int(self.max_iterations * 0.75):
                logging.warning(f"Iteration {iteration} out of {self.max_iterations} max iterations reached during implanting.")
            iteration += 1

        if iteration == self.max_iterations and sum(seqs_per_signal_count.values()) != 0:
            raise RuntimeError(f"{LigoImplanter.__name__}: maximum iterations were reached, but the simulation could not finish "
                               f"with parameters: {vars(self)}.\n")

        repertoires = self._make_repertoire_objects()
        return repertoires

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path):
        raise NotImplementedError

    def _make_background_sequences(self, path) -> bnpdataclass:
        sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{self.seed}_{self.sim_item.name}.tsv"
        make_sequences_from_gen_model(self.sim_item, self.sequence_batch_size, self.seed, sequence_path, self.sequence_type, False)
        return get_bnp_data(sequence_path)

    def _filter_background_sequences(self, sequences: bnpdataclass) -> bnpdataclass:
        annotated_sequences = annotate_sequences(sequences, self.sequence_type == SequenceType.AMINO_ACID, self.all_signals)
        if self.remove_seqs_with_signals:
            annotated_sequences = filter_out_illegal_sequences(annotated_sequences, self.sim_item, self.all_signals, self.max_signals)
        return annotated_sequences

    def _implant_in_sequences(self, sequences: bnpdataclass, seqs_per_signal_count: dict) -> Tuple[bnpdataclass, dict]:

        sequence_lengths = getattr(sequences, self.sequence_type.value).shape.lengths
        remaining_seq_mask = np.ones(len(sequences))
        modified_sequence_dataclass_objs = []

        for signal in self.sim_item.signals:
            if seqs_per_signal_count[signal.id] > 0 and remaining_seq_mask.sum() > 0:
                modified_sequences = []
                motif_instances = self._make_motif_instances(signal, seqs_per_signal_count[signal.id])

                for instance in motif_instances:
                    suitable_seqs = np.argwhere(remaining_seq_mask & sequence_lengths >= len(instance))
                    if len(suitable_seqs) > 0:
                        sequence_index = np.random.choice(suitable_seqs, 1)[0]
                        new_sequence = self._implant_in_sequence(sequences[sequence_index], signal, instance)
                        remaining_seq_mask[sequence_index] = False
                        modified_sequences.append(new_sequence)
                    else:
                        logging.warning(f"{LigoImplanter.__name__}: could not find a sequence to implant {instance} for signal {signal.id}, "
                                        f"skipping for now.")

                modified_sequences = make_bnp_dataclass_from_dicts(modified_sequences)

                if self.sim_item.generative_model.can_compute_p_gens() and (self.export_p_gens or self.keep_p_gen_dist):
                    p_gens = self.sim_item.generative_model.compute_p_gens(modified_sequences, self.sequence_type)
                    modified_sequences = add_field_to_bnp_dataclass(modified_sequences, 'p_gen', float, p_gens)

                modified_sequence_dataclass_objs.append(modified_sequences)

                seqs_per_signal_count[signal.id] -= len(modified_sequences)

        sequences = self._add_info_to_no_signal_sequences(sequences[remaining_seq_mask])

        sequences = merge_dataclass_objects([sequences] + modified_sequence_dataclass_objs)

        return sequences, seqs_per_signal_count

    def _add_info_to_no_signal_sequences(self, sequences) -> bnpdataclass:

        new_fields = {**{s.id: [0 for _ in range(len(sequences))] for s in self.all_signals},
                      **{f"{s.id}_positions": ["m" + "".join("0" for _ in range(sequences.shape.lengths[i])) for i in range(len(sequences))]
                         for s in self.all_signals}}

        if self.sim_item.generative_model.can_compute_p_gens() and (self.export_p_gens or self.keep_p_gen_dist):
            new_fields['p_gen'] = self.sim_item.generative_model.compute_p_gens(sequences, self.sequence_type)

        return add_fields_to_bnp_dataclass(sequences, new_fields)

    def _implant_in_sequence(self, sequence_row: bnpdataclass, signal: Signal, motif_instance: MotifInstance) -> dict:
        imgt_aa_positions = build_imgt_positions(len(sequence_row.get_sequence(self.sequence_type)), motif_instance, sequence_row.region_type)

        limit = len(motif_instance)
        if self.sequence_type == SequenceType.NUCLEOTIDE:
            limit = limit * 3

        position_weights = PositionHelper.build_position_weights(signal.sequence_position_weights, imgt_aa_positions, limit)
        implant_position = choose_implant_position(imgt_aa_positions, position_weights)

        new_sequence = self._make_new_sequence(sequence_row, motif_instance, implant_position)

        new_sequence[signal.id] = 1
        new_sequence[f'{signal.id}_positions'] = "m" + "".join("0" for _ in range(implant_position)) + "1" + \
                                                 "".join("0" for _ in range(len(sequence_row.get_sequence(self.sequence_type)) - implant_position))

        zero_mask = "m" + "".join(["0" for _ in range(getattr(new_sequence, self.sequence_type.value))])
        new_sequence = {**{f"{s.id}_positions": zero_mask for s in self.all_signals}, **new_sequence}
        return new_sequence

    def _make_new_sequence(self, sequence_row: bnpdataclass, motif_instance: MotifInstance, position) -> dict:
        if "/" in motif_instance.instance:
            motif_left, motif_right = motif_instance.instance.split("/")
        else:
            motif_left = motif_instance.instance
            motif_right = ""

        if self.sequence_type == SequenceType.NUCLEOTIDE:
            position *= 3
        sequence_string = getattr(sequence_row, self.sequence_type.value)

        gap_start = position + len(motif_left)
        gap_end = gap_start + motif_instance.gap

        new_sequence_string = sequence_string[:position] + motif_left + sequence_string[gap_start:gap_end] + motif_right + \
                              sequence_string[gap_end + len(motif_right):]

        sequence_dict = {'region_type': sequence_row.region_type, 'frame_type': '', 'v_call': sequence_row.v_call, 'j_call': sequence_row.j_call,
                         'sequence': '', 'sequence_aa': new_sequence_string}

        if self.sequence_type == SequenceType.NUCLEOTIDE:
            sequence_dict['sequence'] = new_sequence_string
            sequence_dict['sequence_aa'] = ReceptorSequence.nt_to_aa(new_sequence_string)

        return sequence_dict

    def _make_motif_instances(self, signal: Signal, seqs_count: int):
        if seqs_count > 0:
            instances = signal.make_motif_instances(seqs_count, self.sequence_type)

            if any(any(not isinstance(el, MotifInstance) for el in signal_motifs) for signal_id, signal_motifs in instances.items()):
                raise NotImplementedError(
                    "When using implanting, V and J genes must not been set in the motifs -- V/J gene implanting is not supported.")

            return instances
        else:
            return []

    def _make_p_gen_histogram(self, sequences: bnpdataclass):
        if not self.sim_item.generative_model.can_compute_p_gens():
            raise RuntimeError(f"{LigoImplanter.__name__}: generative model of class {type(self.sim_item.generative_model).__name__} cannot "
                               f"compute sequence generation probabilities. Use other generative model or set keep_p_gen_dist parameter to False.")

        p_gens = self.sim_item.generative_model.compute_p_gens(sequences, self.sequence_type)
        self.target_p_gen_histogram = np.histogram(np.log10(p_gens), bins=self.p_gen_bin_count, density=True)
        print(self.target_p_gen_histogram)

    def _distribution_matches(self) -> bool:
        raise NotImplementedError

    def _filter_using_p_gens(self, sequences: bnpdataclass, seqs_per_signal_count: dict):
        raise NotImplementedError

    def _make_repertoire_objects(self):
        raise NotImplementedError

    def _make_sequence_paths(self, path: Path):
        tmp_path = PathBuilder.build(path / 'implanted_sequences')
        self.sequence_paths = {signal.id: tmp_path / f'{signal.id}.tsv' for signal in self.sim_item.signals}
        self.sequence_paths['no_signal'] = tmp_path / 'no_signal.tsv'
