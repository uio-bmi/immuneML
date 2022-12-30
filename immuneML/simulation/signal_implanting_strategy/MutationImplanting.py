import copy
import math
from pathlib import Path
from typing import List

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation import SequenceDispenser
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation
from immuneML.simulation.sequence_implanting.SequenceImplantingStrategy import SequenceImplantingStrategy
from immuneML.simulation.signal_implanting_strategy import ImplantingComputation
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class MutationImplanting(SignalImplantingStrategy):
    """
    This class represents a :py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`
    where signals will be implanted in the repertoire by replacing `repertoire_implanting_rate` percent of the sequences with sequences
    generated from the motifs of the signal. Motifs here cannot include gaps and the motif instances are the full sequences and will be
    a part of the repertoire.

    Notes:
        - The sequence type should be IMGT junction, full sequence (or another type where the entire amino acid sequence is used).
        - All Motifs must include V- and J-calls, and mutation position probabilities

    Arguments: this signal implanting strategy has no arguments.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_motif: # cannot include gaps
              seed: CASRSPPVDFGYGYTF # full sequence seed
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
                implanting: Mutation

    """

    # TODO require region type junction or full seq

    def __init__(self, implanting: SequenceImplantingStrategy = None, sequence_position_weights: dict = None,
                 implanting_computation: ImplantingComputation = None):
        super().__init__(implanting, sequence_position_weights, implanting_computation)
        self.sequence_dispenser = None

    def set_sequence_dispenser(self, sequence_dispenser: SequenceDispenser):
        self.sequence_dispenser = sequence_dispenser

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path: Path):

        assert all("/" not in motif.seed for motif in signal.motifs), \
            f"MutationImplanting: motifs cannot include gaps. Check motifs {[motif.identifier for motif in signal.motifs]}."
        assert all(motif.v_call and motif.j_call for motif in signal.motifs), \
            f"MutationImplanting: motifs must have v- and j-calls. Check motifs {[motif.identifier for motif in signal.motifs]}."

        sequences = repertoire.sequences

        new_sequence_count = math.ceil(len(sequences) * repertoire_implanting_rate)
        assert new_sequence_count > 0, \
            f"MutationImplanting: there are too few sequences ({len(sequences)}) in the repertoire with identifier {repertoire.identifier} " \
            f"to have the given repertoire implanting rate ({repertoire_implanting_rate}). Please consider increasing the repertoire implanting rate."
        new_sequences = self._create_new_sequences(sequences, new_sequence_count, signal,
                                                   repertoire_id=repertoire.identifier)
        metadata = copy.deepcopy(repertoire.metadata)
        metadata[signal.id] = True

        for i in sequences:
            i.metadata.custom_params = {}

        return Repertoire.build_from_sequence_objects(new_sequences, path, metadata)

    def _create_new_sequences(self, sequences, new_sequence_count, signal, repertoire_id) -> List[ReceptorSequence]:

        assert self.sequence_dispenser, "No SequenceDispenser has been set"

        new_sequences = sequences[:-new_sequence_count]

        for motif in signal.motifs:
            self.sequence_dispenser.add_seed_sequence(motif)

        for _ in range(new_sequence_count):
            motif = self.sequence_dispenser.generate_mutation(repertoire_id=repertoire_id)

            motif_instance = motif.instantiate_motif()
            annotation = SequenceAnnotation([ImplantAnnotation(signal_id=signal.id, motif_id=motif.identifier,
                                                               motif_instance=motif_instance.instance, position=0)])
            metadata = SequenceMetadata(v_gene=motif.v_call, j_gene=motif.j_call, count=1, chain="B",
                                        region_type=sequences[0].metadata.region_type)

            new_sequences.append(
                ReceptorSequence(amino_acid_sequence=motif_instance.instance, annotation=annotation, metadata=metadata))

        return new_sequences

    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        raise RuntimeError("MutationImplanting was called on a receptor object. Check the simulation parameters.")
