import abc

from source.data_model.sequence.Sequence import Sequence


class SequenceImplantingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def implant(self, sequence: Sequence, signal: dict, sequence_position_weights) -> Sequence:
        pass
