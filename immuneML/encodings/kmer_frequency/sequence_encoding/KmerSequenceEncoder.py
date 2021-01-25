import logging

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from immuneML.util.KmerHelper import KmerHelper


class KmerSequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        """
        creates overlapping continuous k-mers and IMGT position pairs from a sequence as features for use in
        KmerFrequencyEncoder object of type EncoderParams, same object as passed into KmerFrequencyEncoder.
        :param sequence: ReceptorSequence
        :param params: EncoderParams (where params["model"]["k"] is used)
        :return: SequenceEncodingResult
        """
        k = params.model["k"]
        length = len(sequence.get_sequence())

        if length < k:
            logging.warning('KmerSequenceEncoder: Sequence length is less than k. Ignoring sequence...')
            return None

        kmers = KmerHelper.create_kmers_from_sequence(sequence, k)

        return kmers

    @staticmethod
    def get_feature_names(params: EncoderParams):
        return ["sequence"]
