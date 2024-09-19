import abc

from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams


class SequenceEncodingStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_feature_names(params: EncoderParams):
        pass
