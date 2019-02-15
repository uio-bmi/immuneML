from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.util.KmerHelper import KmerHelper


class IMGTGappedKmerEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: dict):

        if sequence.metadata and sequence.metadata.frame_type in ["Out", "Stop"]:
            return None

        k_left = params.get('k_left')
        k_right = params.get('k_right', k_left)
        max_gap = params.get('max_gap')
        min_gap = params.get('min_gap', 0)
        length = len(sequence.get_sequence())

        if length < k_left + k_right + max_gap:
            return None
        gapped_kmers = KmerHelper.create_IMGT_gapped_kmers_from_sequence(sequence, k_left=k_left,
                                                                                    max_gap=max_gap, min_gap=min_gap,
                                                                                    k_right=k_right)
        return gapped_kmers
