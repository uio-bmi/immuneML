import logging

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.KmerHelper import KmerHelper


class IMGTKmerSequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        """
        creates overlapping continuous k-mers from a sequence as features for use in KmerFrequencyEncoder
        object of type EncoderParams, same object as passed into KmerFrequencyEncoder
        :param sequence: ReceptorSequence
        :param params: EncoderParams (where params["model"]["k"] is used)
        :return: SequenceEncodingResult consisting of features and feature information names
        """
        k = params.model["k"]
        sequence_type = params.model.get('sequence_type', EnvironmentSettings.sequence_type)
        length = len(sequence.get_sequence(sequence_type))

        if length < k:
            logging.warning('Sequence length is less than k. Ignoring sequence')
            return None

        kmers = KmerHelper.create_IMGT_kmers_from_sequence(sequence=sequence, k=k, sequence_type=sequence_type)

        kmers = [Constants.FEATURE_DELIMITER.join([str(mer) for mer in kmer]) for kmer in kmers]

        return kmers

    @staticmethod
    def get_feature_names(params: EncoderParams):
        return ["sequence", "imgt_position"]
