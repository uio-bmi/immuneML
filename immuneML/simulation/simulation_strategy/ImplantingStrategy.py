import logging
import random
from dataclasses import fields as get_fields
from typing import List

import numpy as np
from bionumpy.bnpdataclass import BNPDataClass
from olga.utils import nt2aa

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.implants.Signal import Signal, SignalPair
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy
from immuneML.simulation.util.bnp_util import merge_dataclass_objects
from immuneML.simulation.util.util import choose_implant_position, filter_out_illegal_sequences, \
    annotate_sequences
from immuneML.util.PositionHelper import PositionHelper


class ImplantingStrategy(SimulationStrategy):
    MIN_RANGE_PROBABILITY = 1e-5

    def process_sequences(self, sequences: BNPDataClass, seqs_per_signal_count: dict, use_p_gens: bool,
                          sequence_type: SequenceType, sim_item: SimConfigItem, all_signals: List[Signal],
                          remove_positives_first: bool, **kwargs) -> BNPDataClass:

        assert all(not isinstance(signal, SignalPair) for signal in sim_item.signals), \
            f"{ImplantingStrategy.__name__}: 2 signals per sequence are not supported with implanting strategy."

        filtered_sequences = filter_out_illegal_sequences(sequences, sim_item, all_signals,
                                                          max_signals_per_sequence=0 if remove_positives_first else -1,
                                                          max_motifs_per_sequence=1)

        implanting_scaling_factor = kwargs['implanting_scaling_factor'] if 'implanting_scaling_factor' in kwargs else 1

        if len(filtered_sequences) > 0:
            remaining_seq_mask, implanted_sequences = self._implant_in_sequences(filtered_sequences, sequence_type,
                                                                                 sim_item, seqs_per_signal_count,
                                                                                 all_signals, use_p_gens,
                                                                                 implanting_scaling_factor)

            annotated_dc = type(sequences).extend((('original_sequence', str), ('original_p_gen', float)))
            if remaining_seq_mask.sum() > 0:
                added_fields = {'original_sequence': ['' for _ in range(remaining_seq_mask.sum())],
                                'original_p_gen': [-1. for _ in range(remaining_seq_mask.sum())]}
                remaining_seqs = filtered_sequences[remaining_seq_mask].add_fields(added_fields,
                                                                                   field_type_map={
                                                                                       'original_sequence': str,
                                                                                       'original_p_gen': float})
                processed_seqs = merge_dataclass_objects([remaining_seqs, annotated_dc(**implanted_sequences)])
            else:
                processed_seqs = annotated_dc(**implanted_sequences)

            if remove_positives_first and len(processed_seqs) > 0:
                processed_seqs = self._remove_invalid(processed_seqs, sequence_type, sim_item, all_signals,
                                                      annotated_dc)
        else:
            processed_seqs = None

        return processed_seqs

    def _remove_invalid(self, processed_seqs, sequence_type, sim_item, all_signals, annotated_dc):
        processed_seqs = annotate_sequences(processed_seqs, sequence_type == SequenceType.AMINO_ACID, all_signals,
                                            annotated_dc, sim_item.name)
        return filter_out_illegal_sequences(processed_seqs, sim_item, all_signals, max_signals_per_sequence=1,
                                            max_motifs_per_sequence=1)

    def _implant_in_sequences(self, filtered_sequences, sequence_type: SequenceType, sim_item: SimConfigItem,
                              seqs_per_signal_count: dict, all_signals: list, use_p_gens: bool,
                              implanting_scaling_factor: int):

        sequence_lengths = getattr(filtered_sequences, sequence_type.value).lengths
        remaining_seq_mask = np.ones(len(filtered_sequences), dtype=bool)
        implanted_sequences = {**{field.name: [] for field in get_fields(filtered_sequences)},
                               **{'original_sequence': [], 'original_p_gen': []}}

        for signal in sim_item.signals:
            if seqs_per_signal_count[signal.id] > 0 and remaining_seq_mask.sum() > 0:
                motif_instance_count = seqs_per_signal_count[signal.id] * implanting_scaling_factor
                motif_instances = self._make_motif_instances(signal, motif_instance_count, sequence_type)

                for instance in motif_instances:
                    suitable_seqs = np.argwhere(
                        np.logical_and(remaining_seq_mask, sequence_lengths >= len(instance))).reshape(-1)
                    if suitable_seqs.shape[0] > 0:
                        sequence_index = np.random.choice(suitable_seqs, size=1)[0]

                        sequence_obj = self._implant_in_sequence(filtered_sequences[sequence_index], signal, instance,
                                                                 use_p_gens, sequence_type,
                                                                 sim_item, all_signals)

                        if sequence_obj is not None:
                            for field in implanted_sequences.keys():
                                implanted_sequences[field].append(sequence_obj[field])

                            remaining_seq_mask[sequence_index] = False
                            seqs_per_signal_count[signal.id] -= 1

        return remaining_seq_mask, implanted_sequences

    def _implant_in_sequence(self, sequence_row: BackgroundSequences, signal: Signal, motif_instance: MotifInstance,
                             use_p_gens: bool, sequence_type: SequenceType, sim_item: SimConfigItem, all_signals: List[Signal]):

        limit = len(motif_instance)
        if sequence_type == SequenceType.NUCLEOTIDE:
            limit = limit * 3

        sequence_length = len(getattr(sequence_row, sequence_type.value))
        region_type = RegionType[getattr(sequence_row, 'region_type').to_string()]
        position_weights = PositionHelper.get_imgt_position_weights_for_implanting(sequence_length, region_type,
                                                                                   signal.sequence_position_weights,
                                                                                   limit)
        if sum(list(position_weights.values())) == 0:
            logging.info(f"Sequence {sequence_row} has no valid positions where the signal could be implanted, "
                         f"skipping the sequence.")
            return None

        implant_position = choose_implant_position(list(position_weights.keys()), position_weights)

        new_sequence = self._make_new_sequence(sequence_row, motif_instance, implant_position, sequence_type)

        if new_sequence is None:
            return None

        if len(getattr(sequence_row, sequence_type.value)) != len(new_sequence[sequence_type.value]):
            raise RuntimeError(
                f"An error occurred during implanting:\noriginal sequence: {getattr(sequence_row, sequence_type.value).to_string()}\n"
                f"new sequence: {new_sequence[sequence_type.value]}\npositions: {implant_position}\n"
                f"position_weights: {position_weights}\nmotif: {motif_instance.instance}")

        if use_p_gens:
            new_sequence['p_gen'] = sim_item.generative_model.compute_p_gen(
                {key: val.to_string() if hasattr(val, "to_string") else val for key, val in new_sequence.items()},
                sequence_type)
        else:
            new_sequence['p_gen'] = -1.

        new_sequence[signal.id] = 1
        new_sequence[f'{signal.id}_positions'] = "m" + "".join("0" for _ in range(implant_position)) + "1" + \
                                                 "".join("0" for _ in
                                                         range(len(getattr(sequence_row,
                                                                           sequence_type.value)) - implant_position))

        zero_mask = "m" + "".join(["0" for _ in range(len(new_sequence[sequence_type.value]))])
        new_sequence = {**{f"{s.id}_positions": zero_mask for s in all_signals},
                        **{s.id: 0 for s in all_signals if s.id != signal.id},
                        **new_sequence,
                        f'observed_{signal.id}': int(random.uniform(0, 1) > sim_item.false_negative_prob_in_receptors),
                        f'from_default_model': 0, 'duplicate_count': -1, 'signals_aggregated': signal.id,
                        f'original_sequence': getattr(sequence_row, sequence_type.value).to_string(),
                        'original_p_gen': getattr(sequence_row, 'p_gen'),
                        'chain': getattr(sequence_row, 'chain').to_string()}

        return new_sequence

    def _make_new_sequence(self, sequence_row: BackgroundSequences, motif_instance: MotifInstance, position,
                           sequence_type: SequenceType) -> dict:
        if "/" in motif_instance.instance:
            motif_left, motif_right = motif_instance.instance.split("/")
        else:
            motif_left = motif_instance.instance
            motif_right = ""

        if sequence_type == SequenceType.NUCLEOTIDE:
            position *= 3
        sequence_string = getattr(sequence_row, sequence_type.value).to_string()

        gap_start = position + len(motif_left)
        gap_end = gap_start + motif_instance.gap

        new_sequence_string = sequence_string[:position] + motif_left + sequence_string[
                                                                        gap_start:gap_end] + motif_right + \
                              sequence_string[gap_end + len(motif_right):]

        sequence_dict = {'region_type': sequence_row.region_type, 'frame_type': '', 'v_call': sequence_row.v_call,
                         'j_call': sequence_row.j_call,
                         'sequence': '', 'sequence_aa': new_sequence_string}

        if sequence_type == SequenceType.NUCLEOTIDE:
            sequence_dict['sequence'] = new_sequence_string
            sequence_dict['sequence_aa'] = nt2aa(new_sequence_string)
            if "*" in sequence_dict['sequence_aa']:
                sequence_dict = None

        return sequence_dict

    def _make_motif_instances(self, signal: Signal, seqs_count: int, sequence_type: SequenceType):
        if seqs_count > 0:
            instances = signal.make_motif_instances(seqs_count, sequence_type)

            if any(not isinstance(el, MotifInstance) for el in instances):
                raise NotImplementedError(
                    "When using implanting, V and J genes must not been set in the motifs -- V/J gene implanting is not supported.")

            return instances
        else:
            return []
