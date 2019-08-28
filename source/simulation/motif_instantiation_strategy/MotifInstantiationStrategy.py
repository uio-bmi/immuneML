import abc

from source.simulation.implants.MotifInstance import MotifInstance


class MotifInstantiationStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def instantiate_motif(self, base, params: dict = None) -> MotifInstance:
        pass

    @abc.abstractmethod
    def get_max_gap(self) -> int:
        pass
