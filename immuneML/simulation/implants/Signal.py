# quality: gold
from pathlib import Path
from typing import List

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


class Signal:
    """
    This class represents the signal that will be implanted during a Simulation.
    A signal is represented by a list of motifs, and an implanting strategy.

    A signal is associated with a metadata label, which is assigned to a receptor or repertoire.
    For example antigen-specific/disease-associated (receptor) or diseased (repertoire).


    Arguments:

        motifs (list): A list of the motifs associated with this signal.

        implanting (:py:obj:`~immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`):
        The strategy that is used to decide in which sequences the motifs should be implanted, and how. Valid values for this argument are class names of different signal implanting strategies.

        v_gene (str): which V gene is used the create the receptor sequence containing this signal; could be gene name or None to indicate that it
        could be any V gene

        j_gene (str): which J gene is used the create the receptor sequence containing this signal; could be gene name or None to indicate that it
        could be any J gene

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        signals:
            my_signal:
                motifs:
                    - my_simple_motif
                    - my_gapped_motif
                v_gene: V1-1
                j_gene: None
                implanting: HealthySequence
                sequence_position_weights:
                    109: 0.5
                    110: 0.5

    """

    def __init__(self, identifier: str, motifs: List[Motif], implanting_strategy: SignalImplantingStrategy, v_gene: str=None, j_gene: str=None):
        self.id = str(identifier)
        self.motifs = motifs
        self.implanting_strategy = implanting_strategy
        self.v_gene = v_gene
        self.j_gene = j_gene

    def implant_to_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, path: Path) \
            -> Repertoire:
        processed_repertoire = self.implanting_strategy \
            .implant_in_repertoire(repertoire=repertoire,
                                   repertoire_implanting_rate=repertoire_implanting_rate,
                                   signal=self, path=path)
        return processed_repertoire

    def implant_in_sequence(self, sequence: ReceptorSequence, is_noise: bool, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> ReceptorSequence:
        return self.implanting_strategy.implant_in_sequence(sequence=sequence, signal=self, sequence_type=sequence_type)

    def implant_in_receptor(self, receptor: Receptor, is_noise: bool) -> Receptor:
        processed_receptor = self.implanting_strategy.implant_in_receptor(receptor, self, is_noise)
        return processed_receptor

    def __str__(self):
        return "Signal id: " + self.id + "; motifs: " + ", ".join([str(motif) for motif in self.motifs])

    @staticmethod
    def get_documentation():
        initial_doc = str(Signal.__doc__)

        valid_implanting_values = str(
            ReflectionHandler.all_nonabstract_subclass_basic_names(SignalImplantingStrategy, 'Implanting', 'signal_implanting_strategy/'))[
                       1:-1].replace("'", "`")

        docs_mapping = {
            "Valid values for this argument are class names of different signal implanting strategies.":
                f"Valid values are: {valid_implanting_values}"
        }

        doc = update_docs_per_mapping(initial_doc, docs_mapping)
        return doc
