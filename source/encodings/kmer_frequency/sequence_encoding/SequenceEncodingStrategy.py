import abc

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingResult import SequenceEncodingResult


class SequenceEncodingStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams) -> SequenceEncodingResult:
        pass
