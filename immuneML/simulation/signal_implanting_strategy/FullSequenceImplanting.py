import copy
import math
import random
from pathlib import Path
from typing import List

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class FullSequenceImplanting(SignalImplantingStrategy):
    """
    This class represents a :py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`
    where signals will be implanted in the repertoire by replacing `repertoire_implanting_rate` percent of the sequences with sequences
    generated from the motifs of the signal. Motifs here cannot include gaps and the motif instances are the full sequences and will be
    a part of the repertoire.

    Arguments: this signal implanting strategy has no arguments.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_motif: # cannot include gaps
                ...

        signals:
            my_signal:
                motifs:
                    - my_motif
                implanting: FullSequence

    """

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path: Path):

        assert all("/" not in motif.seed for motif in signal.motifs), \
            f'FullSequenceImplanting: motifs cannot include gaps. Check motifs {[motif.identifier for motif in signal.motifs]}.'

        sequences = repertoire.sequences
        new_sequence_count = math.ceil(len(sequences) * repertoire_implanting_rate)
        assert new_sequence_count > 0, \
            f"FullSequenceImplanting: there are too few sequences ({len(sequences)}) in the repertoire with identifier {repertoire.identifier} " \
            f"to have the given repertoire implanting rate ({repertoire_implanting_rate}). Please consider increasing the repertoire implanting rate."
        new_sequences = self._create_new_sequences(sequences, new_sequence_count, signal)
        metadata = copy.deepcopy(repertoire.metadata)
        metadata[f"signal_{signal.id}"] = True

        return Repertoire.build_from_sequence_objects(new_sequences, path, metadata)

    def _create_new_sequences(self, sequences, new_sequence_count, signal) -> List[ReceptorSequence]:
        new_sequences = sequences[:-new_sequence_count]

        for _ in range(new_sequence_count):

            motif = random.choice(signal.motifs)
            motif_instance = motif.instantiate_motif()
            annotation = SequenceAnnotation([ImplantAnnotation(signal_id=signal.id, motif_id=motif.identifier,
                                                               motif_instance=motif_instance.instance, position=0)])
            metadata = SequenceMetadata(v_gene="TRBV6-1", j_gene="TRBJ2-7", count=1, chain="B")

            new_sequences.append(ReceptorSequence(amino_acid_sequence=motif_instance.instance, annotation=annotation, metadata=metadata))

        return new_sequences

    def implant_in_receptor(self, receptor, signal, is_noise: bool):
        raise RuntimeError("FullSequenceImplanting was called on a receptor object. Check the simulation parameters.")
