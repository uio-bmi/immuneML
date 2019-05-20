import warnings

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingResult import SequenceEncodingResult
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.util.KmerHelper import KmerHelper


class KmerSequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams) -> SequenceEncodingResult:
        """
        creates overlapping continuous k-mers and IMGT position pairs from a sequence as features for use in
        KmerFrequencyEncoder object of type EncoderParams, same object as passed into KmerFrequencyEncoder.
        :param sequence: ReceptorSequence
        :param params: EncoderParams (where params["model"]["k"] is used)
        :return: SequenceEncodingResult
        """
        k = params["model"]["k"]
        length = len(sequence.get_sequence())

        if sequence.metadata is not None and sequence.metadata.frame_type.upper() in [SequenceFrameType.OUT.name, SequenceFrameType.STOP.name]:
            warnings.warn('Sequence either has out or stop codon. Ignoring sequence.')
            return SequenceEncodingResult(None, None)

        if length < k:
            warnings.warn('Sequence length is less than k. Ignoring sequence')
            return SequenceEncodingResult(None, None)

        kmers = KmerHelper.create_kmers_from_sequence(sequence, k)

        return SequenceEncodingResult(features=kmers, feature_information_names=["sequence"])
