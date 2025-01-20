import abc
from dataclasses import dataclass
from typing import List

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.MotifInstance import MotifInstance


@dataclass
class Motif:
    '''
    Motifs are the objects which are implanted into sequences during simulation.
    They are defined under :code:`definitions/motifs`. There are several different motif types, each
    having their own parameters.
    '''


    identifier: str

    @abc.abstractmethod
    def get_max_length(self) -> int:
        pass

    @abc.abstractmethod
    def get_alphabet(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get_all_possible_instances(self, sequence_type: SequenceType):
        pass

    @abc.abstractmethod
    def instantiate_motif(self, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> MotifInstance:
        pass
