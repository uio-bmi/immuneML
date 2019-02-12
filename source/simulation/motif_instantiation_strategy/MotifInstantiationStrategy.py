import abc

from source.simulation.implants.MotifInstance import MotifInstance


class MotifInstantiationStrategy(metaclass=abc.ABCMeta):
    # TODO: ensure that all motif instantiation strategies have the same init method signature?

    @abc.abstractmethod
    def instantiate_motif(self, base, params: dict = None) -> MotifInstance:
        pass

    @abc.abstractmethod
    def get_max_gap(self) -> int:
        pass
