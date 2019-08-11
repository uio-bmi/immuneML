import warnings

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.environment.Constants import Constants
from source.util.KmerHelper import KmerHelper


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
        k = params["model"]["k"]
        length = len(sequence.get_sequence())

        if sequence.metadata is not None and sequence.metadata.frame_type.upper() != SequenceFrameType.IN.name:
            warnings.warn('Sequence either has out or stop codon. Ignoring sequence.')
            return None

        if length < k:
            warnings.warn('Sequence length is less than k. Ignoring sequence')
            return None

        kmers = KmerHelper.create_IMGT_kmers_from_sequence(sequence, k)

        kmers = [Constants.FEATURE_DELIMITER.join([str(mer) for mer in kmer]) for kmer in kmers]

        return kmers

    @staticmethod
    def get_feature_names(params: EncoderParams):
        return ["sequence", "imgt_position"]
