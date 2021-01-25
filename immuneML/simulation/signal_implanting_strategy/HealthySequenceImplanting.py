import copy
import logging
import random
from pathlib import Path

from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.sequence_implanting.SequenceImplantingStrategy import SequenceImplantingStrategy
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation, get_implanting_function
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class HealthySequenceImplanting(SignalImplantingStrategy):
    """
    This class represents a :py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`
    where signals will be implanted in 'healthy sequences', meaning sequences in which no signal has been implanted
    previously. This ensures that there is only one signal per receptor sequence.

    If for the given number of sequences in the repertoire and repertoire implanting rate, the total number of sequences for implanting turns out to
    be less than 1 (e.g. for 12000 sequences and repertoire implanting rate 0.00005, it should implant the signal in 0.6 sequences), the signal will
    not be implanted in that repertoire and a warning with repertoire identifier along with the repertoire implanting rate and number of sequences
    in the repertoire will be raised.

    Arguments:

        implanting: name of the implanting strategy, here HealthySequence

        sequence_position_weights (dict): A dictionary describing the relative weights for implanting a signal at each given IMGT position in the
        receptor sequence. If sequence_position_weights are not set, then SequenceImplantingStrategy will make all of the positions equally likely
        for each receptor sequence.

        implanting_computation (str): defines how to determine the number of sequences to implant the signal in a repertoire; it relies on
        repertoire_implanting_rate, but in case where the number of sequences for implanting is not an integer, this option can be useful.
        If implanting rate is set to 'round', then the number of sequences for implanting in a repertoire will be rounded to the nearest integer value of the
        product of repertoire implanting rate and the number of sequences in a repertoire (e.g., if the product value is 1.2, the signal will be
        implanted in one sequence only). If implanting rate is set to 'Poisson', the number of sequences for implanting will be sampled
        from the Poisson distribution with the value of the lambda parameter being repertoire implanting rate multiplied by the number of sequences
        in the repertoire.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_motif:
                ...

        signals:
            my_signal:
                motifs:
                    - my_motif
                    - ...
                implanting: HealthySequence
                implanting_computation: Poisson
                sequence_position_weights:
                    109: 1
                    110: 2
                    111: 5
                    112: 1

    """
    def __init__(self, implanting: SequenceImplantingStrategy = None, sequence_position_weights: dict = None,
                 implanting_computation: ImplantingComputation = None):
        super().__init__(implanting, sequence_position_weights)
        self.compute_implanting = get_implanting_function(implanting_computation)

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path: Path) -> Repertoire:
        max_motif_length = self._calculate_max_motif_length(signal)
        sequences_to_be_processed, other_sequences = self._choose_sequences_for_implanting(repertoire,
                                                                                           repertoire_implanting_rate,
                                                                                           max_motif_length)
        processed_sequences = self._implant_in_sequences(sequences_to_be_processed, signal)
        sequences = other_sequences + processed_sequences
        metadata = self._build_new_metadata(repertoire.metadata, signal)
        new_repertoire = self._build_new_repertoire(sequences, metadata, signal, path)

        return new_repertoire

    def _build_new_metadata(self, metadata: dict, signal) -> dict:
        new_metadata = copy.deepcopy(metadata) if metadata is not None else {}
        new_metadata[f"signal_{signal.id}"] = True
        return new_metadata

    def _calculate_max_motif_length(self, signal):
        max_motif_length = max([motif.get_max_length() for motif in signal.motifs])
        return max_motif_length

    def _build_new_repertoire(self, sequences, repertoire_metadata, signal, path: Path) -> Repertoire:
        if repertoire_metadata is not None:
            metadata = copy.deepcopy(repertoire_metadata)
        else:
            metadata = {}

        # when adding implant to a repertoire, only signal id is stored:
        # more detailed information is available in each receptor_sequence
        # (specific motif and motif instance)
        metadata[f"signal_{signal.id}"] = True
        repertoire = Repertoire.build_from_sequence_objects(sequences, path, metadata)

        return repertoire

    def _implant_in_sequences(self, sequences_to_be_processed: list, signal):
        assert self.sequence_implanting_strategy is not None, \
            "HealthySequenceImplanting: add receptor_sequence implanting strategy when creating a HealthySequenceImplanting object."

        sequences = []
        for sequence in sequences_to_be_processed:
            processed_sequence = self.implant_in_sequence(sequence, signal)
            sequences.append(processed_sequence)

        return sequences

    def _choose_sequences_for_implanting(self, repertoire: Repertoire, repertoire_implanting_rate: float, max_motif_length: int):
        number_of_sequences_to_implant = self.compute_implanting(repertoire_implanting_rate * len(repertoire.sequences))
        if number_of_sequences_to_implant == 0:
            logging.warning(f"HealthySequenceImplanting: there are {len(repertoire.sequences)} sequences in repertoire {repertoire.identifier} "
                            f"for the given repertoire implanting rate of {repertoire_implanting_rate}; no motif will be implanted. To implant "
                            f"motifs, increase 'repertoire_implanting_rate' in the specification.")
        unusable_sequences = []
        unprocessed_sequences = []

        for sequence in repertoire.sequences:
            if sequence.annotation is not None and sequence.annotation.implants is not None and len(sequence.annotation.implants) > 0:
                unusable_sequences.append(sequence)
            elif len(sequence.get_sequence()) < max_motif_length:
                unusable_sequences.append(sequence)
            else:
                unprocessed_sequences.append(sequence)

        assert number_of_sequences_to_implant <= len(unprocessed_sequences), \
            "HealthySequenceImplanting: there are not enough sequences in the repertoire to provide given repertoire infection rate. " \
            f"Reduce repertoire infection rate to proceed. Total unprocessed sequences: {len(unprocessed_sequences)}, " \
            f"number of sequences to implant: {number_of_sequences_to_implant}."

        random.shuffle(unprocessed_sequences)
        sequences_to_be_infected = unprocessed_sequences[:number_of_sequences_to_implant]
        other_sequences = unusable_sequences + unprocessed_sequences[number_of_sequences_to_implant:]

        return sequences_to_be_infected, other_sequences

    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        raise RuntimeError("HealthySequenceImplanting was called on a receptor object. Check the simulation parameters.")
