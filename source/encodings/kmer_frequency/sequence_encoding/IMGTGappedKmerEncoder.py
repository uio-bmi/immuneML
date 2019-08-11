import warnings

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.environment.Constants import Constants
from source.util.KmerHelper import KmerHelper


class IMGTGappedKmerEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        """
        creates all overlapping gapped k-mers and IMGT position pairs from a sequence as features for use in KmerFrequencyEncoder.
        this gap length goes from min_gap to max_gap inclusive, and there is a k-mer of length k_left on the left
        side of the gap and a k-mer of length k_right on the right side of the gap.
        :param sequence: ReceptorSequence
        :param params: EncoderParams (within the "model", the following keys are used: "k_left", "k_right", "max_gap",
                        "min_gap")
        :return: SequenceEncodingResult
        """
        k_left = params["model"].get('k_left')
        k_right = params["model"].get('k_right', k_left)
        max_gap = params["model"].get('max_gap')
        min_gap = params["model"].get('min_gap', 0)
        length = len(sequence.get_sequence())

        if sequence.metadata is not None and sequence.metadata.frame_type.upper() != SequenceFrameType.IN.name:
            warnings.warn('Sequence either has out or stop codon. Ignoring sequence.')
            return None

        if length < k_left + k_right + max_gap:
            warnings.warn('Sequence length is less than k_left + k_right + max_gap. Ignoring sequence')
            return None

        gapped_kmers = KmerHelper.create_IMGT_gapped_kmers_from_sequence(sequence, k_left=k_left, max_gap=max_gap,
                                                                         min_gap=min_gap, k_right=k_right)

        gapped_kmers = [Constants.FEATURE_DELIMITER.join([str(mer) for mer in kmer]) for kmer in gapped_kmers]

        return gapped_kmers

    @staticmethod
    def get_feature_names(params: EncoderParams):
        return ["sequence", "imgt_position"]
