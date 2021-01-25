import abc

from immuneML.simulation.implants.MotifInstance import MotifInstance


class MotifInstantiationStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def instantiate_motif(self, base) -> MotifInstance:
        pass

    @abc.abstractmethod
    def get_max_gap(self) -> int:
        pass
