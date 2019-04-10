import copy
import random

import numpy as np

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.implants.ImplantAnnotation import ImplantAnnotation
from source.simulation.implants.MotifInstance import MotifInstance
from source.simulation.signal_implanting_strategy.sequence_implanting.SequenceImplantingStrategy import \
    SequenceImplantingStrategy
from source.util.PositionHelper import PositionHelper


class GappedMotifImplanting(SequenceImplantingStrategy):

    def implant(self, sequence: ReceptorSequence, signal: dict, sequence_position_weights=None) -> ReceptorSequence:
        motif_instance = signal["motif_instance"]
        imgt_positions = self._build_imgt_positions(sequence, motif_instance)
        limit = len(motif_instance.instance) - motif_instance.instance.count("/") + motif_instance.gap
        position_weights = PositionHelper.build_position_weights(sequence_position_weights, imgt_positions, limit)
        implant_position = self._choose_implant_position(imgt_positions, position_weights)
        new_sequence = self._build_new_sequence(sequence, implant_position, signal)
        return new_sequence

    def _build_imgt_positions(self, sequence: ReceptorSequence, motif_instance: MotifInstance):
        assert len(sequence.get_sequence()) > motif_instance.gap + len(motif_instance.instance) - 1, \
            "The motif instance is longer than receptor_sequence length. Remove the receptor_sequence from the repertoire or reduce max gap length to be able to proceed."
        length = len(sequence.get_sequence())
        return PositionHelper.gen_imgt_positions_from_length(length)

    def _choose_implant_position(self, imgt_positions, position_weights):
        imgt_implant_position = np.random.choice(list(position_weights.keys()), size=1,
                                                 p=list(position_weights.values()))
        possible_indices = np.where(imgt_positions == imgt_implant_position)[0]
        position = random.choice(possible_indices)
        return position

    def _build_new_sequence(self, sequence: ReceptorSequence, position, signal: dict) -> ReceptorSequence:

        gap_length = signal["motif_instance"].gap
        if "/" in signal["motif_instance"].instance:
            motif_left, motif_right = signal["motif_instance"].instance.split("/")
        else:
            motif_left = signal["motif_instance"].instance
            motif_right = ""

        gap_start = position+len(motif_left)
        gap_end = gap_start+gap_length
        part1 = sequence.get_sequence()[:position]
        part2 = sequence.get_sequence()[gap_start:gap_end]
        part3 = sequence.get_sequence()[gap_end+len(motif_right):]

        new_sequence_string = part1 + motif_left + part2 + motif_right + part3

        annotation = SequenceAnnotation()
        implant = ImplantAnnotation(signal_id=signal["signal_id"],
                                    motif_id=signal["motif_id"],
                                    motif_instance=signal["motif_instance"],
                                    position=position)
        annotation.add_implant(implant)

        new_sequence = ReceptorSequence()
        new_sequence.set_annotation(annotation)
        new_sequence.set_metadata(copy.deepcopy(sequence.metadata))
        new_sequence.set_sequence(new_sequence_string, EnvironmentSettings.get_sequence_type())

        return new_sequence
