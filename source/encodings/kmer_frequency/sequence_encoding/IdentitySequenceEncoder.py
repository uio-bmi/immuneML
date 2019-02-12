from source.data_model.sequence.Sequence import Sequence
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy


class IdentitySequenceEncoder(SequenceEncodingStrategy):
    """
    Allows to measure the frequency of the sequence in the dataset
    """

    @staticmethod
    def encode_sequence(sequence: Sequence, params: dict):

        encoded = sequence.get_sequence()
        res = [encoded]

        if sequence.metadata is not None and sequence.metadata.frame_type in ["Out", "Stop"]:
            res = None

        return res
