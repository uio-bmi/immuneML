from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy


class IdentitySequenceEncoder(SequenceEncodingStrategy):
    """
    Allows to measure the frequency of the receptor_sequence in the dataset
    """

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):

        encoded = sequence.get_sequence()
        res = [encoded]

        if sequence.metadata is not None and sequence.metadata.frame_type in ["Out", "Stop"]:
            res = None

        return res
