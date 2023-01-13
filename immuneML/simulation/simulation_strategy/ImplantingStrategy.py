import logging
import random
from dataclasses import fields as get_fields

import numpy as np
from bionumpy.bnpdataclass import BNPDataClass

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy
from immuneML.simulation.util.bnp_util import merge_dataclass_objects
from immuneML.simulation.util.util import build_imgt_positions, choose_implant_position
from immuneML.util.PositionHelper import PositionHelper


class ImplantingStrategy(SimulationStrategy):

    MIN_RANGE_PROBABILITY = 1e-5

    def process_sequences(self, sequences: BNPDataClass, seqs_per_signal_count: dict, max_signals: int, use_p_gens: bool, sequence_type: SequenceType,
                          sim_item: SimConfigItem) \
        -> BNPDataClass:

        sequence_lengths = getattr(sequences, sequence_type.value).lengths
        remaining_seq_mask = np.ones(len(sequences), dtype=bool)
        implanted_sequences = {field.name: [] for field in get_fields(sequences)}

        for signal in sim_item.signals:
            if seqs_per_signal_count[signal.id] > 0 and remaining_seq_mask.sum() > 0:
                motif_instances = self._make_motif_instances(signal, seqs_per_signal_count[signal.id])

                for instance in motif_instances:
                    suitable_seqs = np.argwhere(np.logical_and(remaining_seq_mask, sequence_lengths >= len(instance))).reshape(-1)
                    if suitable_seqs.shape[0] > 0:
                        sequence_index = np.random.choice(suitable_seqs, size=1)[0]

                        sequence_obj = self._implant_in_sequence(sequences[sequence_index], signal, instance, use_p_gens)
                        for field in get_fields(sequences):
                            implanted_sequences[field.name].append(sequence_obj[field.name])

                        remaining_seq_mask[sequence_index] = False
                        seqs_per_signal_count[signal.id] -= 1
                    else:
                        logging.warning(f"{ImplantingStrategy.__name__}: could not find a sequence to implant {instance} for signal {signal.id}, "
                                        f"skipping for now.")

        return merge_dataclass_objects([sequences[remaining_seq_mask], type(sequences)(**implanted_sequences)])



    def _implant_in_sequence(self, sequence_row: BackgroundSequences, signal: Signal, motif_instance: MotifInstance, use_p_gens: bool) -> dict:
        imgt_aa_positions = build_imgt_positions(len(getattr(sequence_row, self._state.sequence_type.value)), motif_instance,
                                                 sequence_row.region_type)

        limit = len(motif_instance)
        if self._state.sequence_type == SequenceType.NUCLEOTIDE:
            limit = limit * 3

        position_weights = PositionHelper.build_position_weights(signal.sequence_position_weights, imgt_aa_positions, limit)
        implant_position = choose_implant_position(imgt_aa_positions, position_weights)

        new_sequence = self._make_new_sequence(sequence_row, motif_instance, implant_position)

        if use_p_gens:
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
        new_sequence = {**{f"{s.id}_positions": zero_mask for s in self._state.all_signals}, **new_sequence,
                        f'observed_{signal.id}': int(random.uniform(0, 1) > self._state.sim_item.false_negative_prob_in_receptors)}

        return new_sequence

    def _make_new_sequence(self, sequence_row: BackgroundSequences, motif_instance: MotifInstance, position) -> dict:
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
