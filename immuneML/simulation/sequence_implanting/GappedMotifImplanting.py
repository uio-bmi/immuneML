import copy

import numpy as np

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.sequence_implanting.SequenceImplantingStrategy import SequenceImplantingStrategy
from immuneML.util.PositionHelper import PositionHelper


class GappedMotifImplanting(SequenceImplantingStrategy):

    def implant(self, sequence: ReceptorSequence, signal: dict, sequence_position_weights=None, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> ReceptorSequence:

        assert sequence.metadata.region_type == RegionType.IMGT_CDR3, \
            f"{GappedMotifImplanting.__name__}: sequence is of type {sequence.metadata.region_type}, but currently only {RegionType.IMGT_CDR3} " \
            f"is supported."

        motif_instance = signal["motif_instance"]
        imgt_aa_positions = self._build_imgt_positions(sequence, motif_instance, sequence_type)

        limit = len(motif_instance.instance) - motif_instance.instance.count("/") + motif_instance.gap - 1
        if sequence_type == SequenceType.NUCLEOTIDE:
            limit = limit * 3

        position_weights = PositionHelper.build_position_weights(sequence_position_weights, imgt_aa_positions, limit)
        implant_position = self._choose_implant_position(imgt_aa_positions, position_weights)
        new_sequence = self._build_new_sequence(sequence, implant_position, signal, sequence_type)
        return new_sequence

    def _build_imgt_positions(self, sequence: ReceptorSequence, motif_instance: MotifInstance, sequence_type: SequenceType):
        assert len(sequence.get_sequence(sequence_type)) >= motif_instance.gap + len(motif_instance.instance) - 1, \
            "The motif instance is longer than receptor_sequence length. Remove the receptor_sequence from the repertoire or reduce max gap length " \
            "to be able to proceed. "
        return PositionHelper.gen_imgt_positions_from_sequence(sequence)

    def _choose_implant_position(self, imgt_positions, position_weights):
        imgt_implant_position = np.random.choice(list(position_weights.keys()), size=1, p=list(position_weights.values()))
        position = np.where(imgt_positions == imgt_implant_position)[0][0]
        return position

    def _build_new_sequence(self, sequence: ReceptorSequence, position, signal: dict, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> ReceptorSequence:

        gap_length = signal["motif_instance"].gap
        if "/" in signal["motif_instance"].instance:
            motif_left, motif_right = signal["motif_instance"].instance.split("/")
        else:
            motif_left = signal["motif_instance"].instance
            motif_right = ""

        if sequence_type == SequenceType.NUCLEOTIDE:
            position *= 3

        gap_start = position+len(motif_left)
        gap_end = gap_start+gap_length
        part1 = sequence.get_sequence(sequence_type)[:position]
        part2 = sequence.get_sequence(sequence_type)[gap_start:gap_end]
        part3 = sequence.get_sequence(sequence_type)[gap_end+len(motif_right):]

        new_sequence_string = part1 + motif_left + part2 + motif_right + part3

        annotation = SequenceAnnotation()
        annotation.add_implant(ImplantAnnotation(signal_id=signal["signal_id"], motif_id=signal["motif_id"], motif_instance=signal["motif_instance"],
                                                 position=position))

        new_sequence = ReceptorSequence(annotation=annotation, metadata=copy.deepcopy(sequence.metadata))
        new_sequence.metadata.custom_params[signal['signal_id']] = True
        new_sequence.set_sequence(new_sequence_string, sequence_type)

        return new_sequence
