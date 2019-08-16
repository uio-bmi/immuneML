import abc

from source.data_model.receptor.receptor_sequence import ReceptorSequence


class SequenceImplantingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def implant(self, sequence: ReceptorSequence, signal: dict, sequence_position_weights) -> ReceptorSequence:
        pass
