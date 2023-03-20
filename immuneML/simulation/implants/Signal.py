import random
from dataclasses import dataclass
from typing import List, Union

from immuneML import Constants
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.MotifInstance import MotifInstanceGroup
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

        motifs (list): A list of the motifs associated with this signal, either defined by seed or by position weight matrix. Alternatively, it can be a list of a list of motifs, in which case the motifs in the same sublist (max 2 motifs) have to co-occur in the same sequence

        sequence_position_weights (dict): a dictionary specifying for each IMGT position in the sequence how likely it is for signal to be there. For positions not specified, the probability of having the signal there is 0.

        v_call (str): V gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;

        j_call (str): J gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;

        clonal_frequency (dict): clonal frequency in Ligo is simulated through `scipy's zeta distribution function for generating random numbers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zipf.html>`_, with parameters provided under clonal_frequency parameter. If clonal frequency shouldn't be used, this parameter can be None

            .. indent with spaces
            .. code-block:: yaml

                clonal_frequency:
                    a: 2 # shape parameter of the distribution
                    loc: 0 # 0 by default but can be used to shift the distribution


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
                v_call: TRBV1
                j_call: TRBJ1
                clonal_frequency:
                    a: 2
                    loc: 0

    """
    id: str
    motifs: List[Union[Motif, List[Motif]]]
    sequence_position_weights: dict = None
    v_call: str = None
    j_call: str = None
    clonal_frequency: dict = None

    def get_all_motif_instances(self, sequence_type: SequenceType):
        motif_instances = []
        for motif_group in self.motifs:
            if isinstance(motif_group, list):
                motif_instances.append(MotifInstanceGroup([motif.get_all_possible_instances(sequence_type) for motif in motif_group]))
            else:
                motif_instances.append(motif_group.get_all_possible_instances(sequence_type))
        return motif_instances

    def make_motif_instances(self, count, sequence_type: SequenceType):
        return [motif.instantiate_motif(sequence_type=sequence_type) for motif in random.choices(self.motifs, k=count)]

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(vars(self))

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


@dataclass
class SignalPair:
    signal1: Signal
    signal2: Signal

    @property
    def id(self) -> str:
        return Constants.SIGNAL_DELIMITER.join(sorted([self.signal1.id, self.signal2.id]))

    @property
    def v_call(self):
        return [el for el in sorted(list({self.signal1.v_call, self.signal2.v_call})) if el is not None] \
            if self.signal1.v_call is not None and self.signal2.v_call is not None else None

    @property
    def j_call(self):
        return [el for el in sorted(list({self.signal1.j_call, self.signal2.j_call})) if el is not None] \
            if self.signal1.j_call is not None and self.signal2.j_call is not None else None

    @property
    def clonal_frequency(self):
        return random.choice([self.signal1, self.signal2]).clonal_frequency

    def __hash__(self):
        return hash(tuple(sorted([self.signal1.id, self.signal2.id])))
