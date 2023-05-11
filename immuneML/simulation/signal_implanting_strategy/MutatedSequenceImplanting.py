import copy
import math
from pathlib import Path
import random
from typing import List

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation import SequenceDispenser
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation
from immuneML.simulation.sequence_implanting.SequenceImplantingStrategy import SequenceImplantingStrategy
from immuneML.simulation.signal_implanting_strategy import ImplantingComputation
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class MutatedSequenceImplanting(SignalImplantingStrategy):
    """
    This class represents a :py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`
    where signals will be implanted in the repertoire by appending or replacing `repertoire_implanting_rate` percent of the sequences with mutated
    sequences that are generated from the full sequences from the signal.

    Notes:
        - The sequence type must be IMGT junction
        - All Motifs must include V- and J-calls, and mutation position probabilities

    Arguments:

        implanting: name of the implanting strategy, here Decoy

        mutation_hamming_distance (int): The number of positions to mutate

        occurrence_limit_pgen_range (dict): The max limit of occurrences for implanted mutated sequences based on their generation probability

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_motif: # cannot include gaps
              seed: CASRSPPVDFGYGYTF # full seed sequence
              v_call: TRBV10-1
              j_call: TRBJ1-2
              mutation_position_possibilities: # sum must be 1
                6: 0.1
                7: 0.2
                8: 0.4
                9: 0.2
                10: 0.1



        signals:
            my_signal:
                motifs:
                    - my_motif
                implanting: MutatedSequence
                    mutation_hamming_distance: 2
                    occurrence_limit_pgen_range:
                        1.01e-11: 2
                        1.67e-09: 4
                        1.76e-08: 5
                        2.35e-08: 6

    """

    def __init__(self, implanting: SequenceImplantingStrategy = None, sequence_position_weights: dict = None,
                 implanting_computation: ImplantingComputation = None, mutation_hamming_distance: int = 1,
                 occurrence_limit_pgen_range: dict = None, overwrite_sequences: bool = False,
                 nr_of_decoys: int = None, dataset_implanting_rate_per_decoy: float = None,
                 repertoire_implanting_rate_per_decoy: float = None):
        super().__init__(implanting, sequence_position_weights, implanting_computation, mutation_hamming_distance,
                         occurrence_limit_pgen_range, overwrite_sequences, nr_of_decoys,
                         dataset_implanting_rate_per_decoy, repertoire_implanting_rate_per_decoy)
        self.sequence_dispenser = None

    def set_sequence_dispenser(self, sequence_dispenser: SequenceDispenser):
        self.sequence_dispenser = sequence_dispenser

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path: Path):

        assert repertoire.get_region_type() == RegionType.IMGT_JUNCTION, \
            f"MutatedSequenceImplanting: RegionType must be IMGT_Junction, not {repertoire.get_region_type()}"
        assert all("/" not in motif.seed for motif in signal.motifs), \
            f"MutatedSequenceImplanting: motifs cannot include gaps. Check motifs {[motif.identifier for motif in signal.motifs]}."
        assert all(motif.v_call and motif.j_call for motif in signal.motifs), \
            f"MutatedSequenceImplanting: motifs must have v- and j-calls. Check motifs {[motif.identifier for motif in signal.motifs]}."

        sequences = repertoire.sequences

        new_sequence_count = math.ceil(len(sequences) * repertoire_implanting_rate)
        assert new_sequence_count > 0, \
            f"MutatedSequenceImplanting: there are too few sequences ({len(sequences)}) in the repertoire with identifier {repertoire.identifier} " \
            f"to have the given repertoire implanting rate ({repertoire_implanting_rate}). Please consider increasing the repertoire implanting rate."

        new_sequences = self._create_new_sequences(sequences, new_sequence_count, signal, repertoire)

        metadata = copy.deepcopy(repertoire.metadata)
        metadata[signal.id] = True

        for i in sequences:
            i.metadata.custom_params = {}

        return Repertoire.build_from_sequence_objects(new_sequences, path, metadata)

    def _create_new_sequences(self, sequences, new_sequence_count, signal, repertoire) -> List[ReceptorSequence]:

        assert self.sequence_dispenser, "No SequenceDispenser has been set"

        if self.overwrite_sequences:
            random.shuffle(sequences)
            new_sequences = sequences[:-new_sequence_count]
        else:
            new_sequences = sequences

        for motif in signal.motifs:
            self.sequence_dispenser.add_seed_sequence(motif)

        for _ in range(new_sequence_count):
            motif = self.sequence_dispenser.generate_mutation(repertoire_id=repertoire.identifier)
            motif_instance = motif.instantiate_motif()

            seq_count = self._draw_random_count(motif_instance.instance, repertoire.get_counts())
            self.sequence_dispenser.append_count_to_seq_occurrences(motif_instance.instance, seq_count, repertoire.identifier)

            annotation = SequenceAnnotation([ImplantAnnotation(signal_id=signal.id, motif_id=motif.identifier,
                                                               motif_instance=motif_instance.instance, position=0)])
            metadata = SequenceMetadata(v_gene=motif.v_call, j_gene=motif.j_call,
                                        count=seq_count,
                                        # TODO get chain and account for maybe no chain. use repertoire.get_chains()[0],
                                        chain="B",
                                        region_type=sequences[0].metadata.region_type)

            new_sequences.append(
                ReceptorSequence(amino_acid_sequence=motif_instance.instance, annotation=annotation, metadata=metadata))

        return new_sequences

    def _draw_random_count(self, seq: str, sequence_counts):
        random_count = random.choice(sequence_counts)

        variance = random_count / 2

        count = round(random.gauss(random_count, variance))

        if count < 1:
            count = 1

        occurrence_limit = self.sequence_dispenser.get_occurrence_limit(seq)
        num_of_occurrences = self.sequence_dispenser.get_seq_occurrences(seq)

        if count + num_of_occurrences > occurrence_limit:
            count = occurrence_limit-num_of_occurrences

        return count

    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        raise RuntimeError(
            "MutatedSequenceImplanting was called on a receptor object. Check the simulation parameters.")
