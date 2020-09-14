# quality: gold

import abc
import random

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.simulation.sequence_implanting.SequenceImplantingStrategy import SequenceImplantingStrategy


class SignalImplantingStrategy(metaclass=abc.ABCMeta):

    def __init__(self, implanting: SequenceImplantingStrategy = None, sequence_position_weights: dict = None):
        self.sequence_implanting_strategy = implanting
        self.sequence_position_weights = sequence_position_weights

    @abc.abstractmethod
    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path):
        pass

    def implant_in_sequence(self, sequence: ReceptorSequence, signal, motif=None, chain=None) -> ReceptorSequence:
        assert self.sequence_implanting_strategy is not None, \
            "SignalImplanting: set SequenceImplantingStrategy in SignalImplanting object before calling implant_in_sequence method."

        if motif is None:
            motif = random.choice(signal.motifs)

        motif_instance = motif.instantiate_motif(chain)
        new_sequence = self.sequence_implanting_strategy.implant(sequence=sequence,
                                                                 signal={"signal_id": signal.id,
                                                                         "motif_id": motif.identifier,
                                                                         "motif_instance": motif_instance},
                                                                 sequence_position_weights=self.sequence_position_weights)
        return new_sequence

    @abc.abstractmethod
    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        pass
