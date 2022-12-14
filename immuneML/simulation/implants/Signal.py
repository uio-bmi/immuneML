# quality: gold
import random
from dataclasses import dataclass
from itertools import chain
from typing import List

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.signal_implanting.SignalImplantingStrategy import SignalImplantingStrategy
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


@dataclass
class Signal:
    """
    This class represents the signal that will be implanted during a Simulation.
    A signal is represented by a list of motifs, and optionally, positions weights showing where one of the motifs of the signal can
    occur in a sequence.

    A signal is associated with a metadata label, which is assigned to a receptor or repertoire.
    For example antigen-specific/disease-associated (receptor) or diseased (repertoire).


    Arguments:

        motifs (list): A list of the motifs associated with this signal.

        sequence_position_weights (dict): a dictionary specifying for each IMGT position in the sequence how likely it is for signal to be there. For positions not specified, the probability of having the signal there is 0.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        signals:
            my_signal:
                motifs:
                    - my_simple_motif
                    - my_gapped_motif
                sequence_position_weights:
                    109: 0.5
                    110: 0.5

    """
    id: str
    motifs: List[Motif]
    sequence_position_weights: dict = None

    def is_in(self, sequence: dict, sequence_type: SequenceType):
        return any(motif.is_in(sequence, sequence_type) for motif in self.motifs)

    def get_all_motif_instances(self, sequence_type: SequenceType):
        return chain((motif.get_all_possible_instances(sequence_type), motif.v_call, motif.j_call) for motif in self.motifs)

    def make_motif_instances(self, count, sequence_type):
        return [motif.instantiate_motif(sequence_type=sequence_type) for motif in random.choices(self.motifs, k=count)]

    def __str__(self):
        return "Signal id: " + self.id + "; motifs: " + ", ".join([str(motif) for motif in self.motifs])

    @staticmethod
    def get_documentation():
        initial_doc = str(Signal.__doc__)

        valid_implanting_values = str(
            ReflectionHandler.all_nonabstract_subclass_basic_names(SignalImplantingStrategy, 'Implanting', 'signal_implanting/'))[
                                  1:-1].replace("'", "`")

        docs_mapping = {
            "Valid values for this argument are class names of different signal implanting strategies.":
                f"Valid values are: {valid_implanting_values}"
        }

        doc = update_docs_per_mapping(initial_doc, docs_mapping)
        return doc
