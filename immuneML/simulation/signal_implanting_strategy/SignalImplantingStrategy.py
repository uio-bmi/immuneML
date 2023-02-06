# quality: gold

import abc
import random
from pathlib import Path

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.sequence_implanting.SequenceImplantingStrategy import SequenceImplantingStrategy
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation


class SignalImplantingStrategy(metaclass=abc.ABCMeta):

    def __init__(self, implanting: SequenceImplantingStrategy = None, sequence_position_weights: dict = None,
                 implanting_computation: ImplantingComputation = None, mutation_hamming_distance: int = 1,
                 occurrence_limit_pgen_range: dict = None, overwrite_sequences: bool = False,
                 nr_of_decoys: int = None, dataset_implanting_rate_per_decoy: float = None, repertoire_implanting_rate_per_decoy: float = None):
        self.sequence_implanting_strategy = implanting
        self.sequence_position_weights = sequence_position_weights
        self.compute_implanting = implanting_computation

        self.overwrite_sequences = overwrite_sequences

        self.mutation_hamming_distance = mutation_hamming_distance
        self.occurrence_limit_pgen_range = occurrence_limit_pgen_range

        self.nr_of_decoys = nr_of_decoys
        self.dataset_implanting_rate_per_decoy = dataset_implanting_rate_per_decoy
        self.repertoire_implanting_rate_per_decoy = repertoire_implanting_rate_per_decoy

    @abc.abstractmethod
    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path: Path):
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
