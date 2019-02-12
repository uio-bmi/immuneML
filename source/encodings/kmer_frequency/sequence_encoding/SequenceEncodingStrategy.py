import abc

from source.data_model.sequence.Sequence import Sequence


class SequenceEncodingStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def encode_sequence(sequence: Sequence, params: dict):
        pass
