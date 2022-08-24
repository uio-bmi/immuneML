import abc

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.MotifInstance import MotifInstance


class MotifInstantiationStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def instantiate_motif(self, base, sequence_type: SequenceType = SequenceType.AMINO_ACID) -> MotifInstance:
        pass

    @abc.abstractmethod
    def get_max_gap(self) -> int:
        pass

    @abc.abstractmethod
    def get_all_possible_instances(self, base: str, sequence_type: SequenceType) -> list:
        pass
