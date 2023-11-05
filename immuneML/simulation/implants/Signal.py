import random
import typing
from dataclasses import dataclass
from typing import List, Union

from immuneML import Constants
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.MotifInstance import MotifInstanceGroup


@dataclass
class Signal:
    """
    This class represents the simulated signal.
    A signal is represented by a list of motifs, and optionally, position weights showing where one of the motifs of the signal can
    occur in a sequence.

    A signal is associated with a metadata label, which is assigned to a receptor or repertoire.
    For example antigen-specific/disease-associated (receptor) or diseased (repertoire).

    .. note:: IMGT positions

        To use sequence position weights, IMGT positions should be explicitly specified as strings, under quotation marks, to allow for all positions to be properly distinguished.

    Specification arguments:

    - motifs (list): A list of the motifs associated with this signal, either defined by seed or by position weight matrix. Alternatively, it can be a list of a list of motifs, in which case the motifs in the same sublist (max 2 motifs) have to co-occur in the same sequence

    - sequence_position_weights (dict): a dictionary specifying for each IMGT position in the sequence how likely it is for the signal to be there. If the position is not present in the sequence, the probability of the signal occurring at that position will be redistributed to other positions with probabilities that are not explicitly set to 0 by the user.

    - v_call (str): V gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;

    - j_call (str): J gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;

    - source_file (str): path to the file where the custom signal function is; cannot be combined with the arguments listed above (motifs, v_call, j_call, sequence_position_weights)

    - is_present_func (str): name of the function from the source_file file that will be used to specify the signal; the function's signature must be:

    .. code-block:: python

        def is_present(sequence_aa: str, sequence: str, v_call: str, j_call: str) -> bool:
            # custom implementation where all or some of these arguments can be used

    - clonal_frequency (dict): clonal frequency in Ligo is simulated through `scipy's zeta distribution function for generating random numbers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zipf.html>`_, with parameters provided under clonal_frequency parameter. If clonal frequency should not be used, this parameter can be None

    .. code-block:: yaml

      clonal_frequency:
        a: 2 # shape parameter of the distribution
        loc: 0 # 0 by default but can be used to shift the distribution


    YAML specification:

    .. code-block:: yaml

        signals:
            my_signal:
                motifs:
                    - my_simple_motif
                    - my_gapped_motif
                sequence_position_weights:
                    '109': 0.5
                    '110': 0.5
                v_call: TRBV1
                j_call: TRBJ1
                clonal_frequency:
                    a: 2
                    loc: 0
            signal_with_custom_func:
                source_file: signal_func.py
                is_present_func: is_signal_present
                clonal_frequency:
                    a: 2
                    loc: 0

    """
    id: str
    motifs: List[Union[Motif, List[Motif]]] = None
    sequence_position_weights: dict = None
    v_call: str = None
    j_call: str = None
    clonal_frequency: dict = None
    is_present_custom_func: typing.Callable = None

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
