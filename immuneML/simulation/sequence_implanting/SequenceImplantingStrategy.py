import abc

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class SequenceImplantingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def implant(self, sequence: ReceptorSequence, signal: dict, sequence_position_weights) -> ReceptorSequence:
        pass
