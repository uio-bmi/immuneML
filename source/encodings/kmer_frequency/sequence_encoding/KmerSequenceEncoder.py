from source.data_model.sequence.Sequence import Sequence
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.util.KmerHelper import KmerHelper


class KmerSequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: Sequence, params: dict):
        k = params["k"]
        length = len(sequence.get_sequence())
        if length < k:
            raise ValueError('Sequence length is less than k. '
                             'Filter sequences from each repertoire that are less than length k then rerun.')
        kmers = KmerHelper.create_kmers_from_sequence(sequence, k)
        return kmers
