import logging

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.KmerHelper import KmerHelper


class KmerSequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        """
        Encodes a receptor sequence into a sequence of k-mers

        Args:
            sequence: ReceptorSequence object
            params: EncoderParams object with information on k-mer length

        Returns:

        """
        k = params.model["k"]
        sequence_type = params.model.get('sequence_type', EnvironmentSettings.sequence_type)
        length = len(sequence.get_sequence(sequence_type))

        if length < k:
            logging.warning(f'KmerSequenceEncoder: Sequence length {length} is less than {k}. Ignoring sequence...')
            return None

        kmers = KmerHelper.create_kmers_from_sequence(sequence=sequence, k=k, sequence_type=sequence_type)

        return kmers

    @staticmethod
    def get_feature_names(params: EncoderParams):
        return ["sequence"]
