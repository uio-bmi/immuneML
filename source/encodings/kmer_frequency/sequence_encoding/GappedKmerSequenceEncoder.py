from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.util.KmerHelper import KmerHelper


class GappedKmerSequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: dict):
        k_left = params.get('k_left')
        k_right = params.get('k_right', k_left)
        max_gap = params.get('max_gap')
        min_gap = params.get('min_gap', 0)
        length = len(sequence.get_sequence())
        if length < k_left + k_right + max_gap:
            raise ValueError('Sequence length is less than k_left + k_right + max_gap. '
                             'Filter sequences from each repertoire that are less than this length then rerun.')
        gapped_kmers = KmerHelper.create_gapped_kmers_from_sequence(sequence, k_left=k_left, max_gap=max_gap,
                                                                      min_gap=min_gap, k_right=k_right)
        return gapped_kmers
