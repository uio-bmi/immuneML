import copy
import random

from source.data_model.repertoire.Repertoire import Repertoire
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class HealthySequenceImplanting(SignalImplantingStrategy):
    """
    This class represents a :py:obj:`~source.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`
    where signals will be implanted in 'healthy sequences', meaning sequences in which no signal has been implanted
    previously. This ensures that there is only one signal per receptor sequence.

    Arguments:

        implanting: name of the implanting strategy, here HealthySequence

        sequence_position_weights (dict): A dictionary describing the relative weights for implanting a signal
            at each given IMGT position in the receptor sequence. If sequence_position_weights are not set,
            then SequenceImplantingStrategy will make all of the positions equally likely for each receptor sequence.


    Specification:

        motifs:
            my_motif:
                ...

        signals:
            my_signal:
                motifs:
                    - my_motif
                    - ...
                implanting: HealthySequence
                sequence_position_weights:
                    109: 1
                    110: 2
                    111: 5
                    112: 1

    """

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path) -> Repertoire:
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

    def _build_new_repertoire(self, sequences, repertoire_metadata, signal, path) -> Repertoire:
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
        number_of_sequences_to_implant = int(repertoire_implanting_rate * len(repertoire.sequences))
        unusable_sequences = []
        unprocessed_sequences = []

        for sequence in repertoire.sequences:
            if sequence.annotation is not None and sequence.annotation.implants is not None and len(sequence.annotation.implants) > 0:
                unusable_sequences.append(sequence)
            elif len(sequence.get_sequence()) <= max_motif_length:
                unusable_sequences.append(sequence)
            else:
                unprocessed_sequences.append(sequence)

        assert number_of_sequences_to_implant <= len(unprocessed_sequences), \
            "HealthySequenceImplanting: there are not enough sequences in the repertoire to provide given repertoire infection rate. " \
            "Reduce repertoire infection rate to proceed."

        random.shuffle(unprocessed_sequences)
        sequences_to_be_infected = unprocessed_sequences[:number_of_sequences_to_implant]
        other_sequences = unusable_sequences + unprocessed_sequences[number_of_sequences_to_implant:]

        return sequences_to_be_infected, other_sequences

    def implant_in_receptor(self, receptor, signal):
        raise RuntimeError("HealthySequenceImplanting was called on a receptor object. Check the simulation parameters.")
