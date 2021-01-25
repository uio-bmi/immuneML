import warnings

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from immuneML.util.KmerHelper import KmerHelper


class GappedKmerSequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        """
        creates all overlapping gapped k-mers from a sequence as features for use in KmerFrequencyEncoder.
        this gap length goes from min_gap to max_gap inclusive, and there is a k-mer of length k_left on the left
        side of the gap and a k-mer of length k_right on the right side of the gap.
        :param sequence: ReceptorSequence
        :param params: EncoderParams (within the "model", the following keys are used: "k_left", "k_right", "max_gap",
                        "min_gap")
        :return: SequenceEncodingResult
        """
        k_left = params.model.get('k_left')
        k_right = params.model.get('k_right', k_left)
        max_gap = params.model.get('max_gap')
        min_gap = params.model.get('min_gap', 0)
        length = len(sequence.get_sequence())

        if length < k_left + k_right + max_gap:
            warnings.warn('Sequence length is less than k_left + k_right + max_gap. Ignoring sequence')
            return None

        gapped_kmers = KmerHelper.create_gapped_kmers_from_sequence(sequence, k_left=k_left, max_gap=max_gap,
                                                                    min_gap=min_gap, k_right=k_right)

        return gapped_kmers

    @staticmethod
    def get_feature_names(params: EncoderParams):
        return ["sequence"]
