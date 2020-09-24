from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFreqRepertoireEncoder import KmerFreqRepertoireEncoder
from source.util.ParameterValidator import ParameterValidator


class KmerPresenceEncoder(KmerFreqRepertoireEncoder):
    """
        The KmerPresenceEncoder class encodes a repertoire by the presence/absence of k-mers in all of the sequences of that
        repertoire. A k-mer is a sequence of letters of length k into which an immune receptor sequence can be decomposed.
        K-mers can be defined in different ways, as determined by the sequence_encoding. This encoder is a variation of
        KmerFrequencyEncoder - KmerFrequencyEncoder encodes k-mer frequencies, whereas KmerPresenceEncoder encodes whether
        the k-mer is present/absent in the repertoire. Note that KmerPresenceEncoder currently supports only RepertoireDataset.


        Attributes:
            sequence_encoding (:py:mod:`source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType`):
                The type of k-mers that are used. The simplest sequence_encoding is :py:mod:`source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.CONTINUOUS_KMER`,
                which simply uses contiguous subsequences of length k to represent the k-mers.
                When gapped k-mers are used (:py:mod:`source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`,
                :py:mod:`source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`), the k-mers may contain
                gaps with a size between min_gap and max_gap, and the k-mer length is defined as a combination of k_left and k_right.
                When IMGT k-mers are used (:py:mod:`source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_CONTINUOUS_KMER`,
                :py:mod:`source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_GAPPED_KMER`), IMGT positional information is
                taken into account (i.e. the same sequence in a different position is considered to be a different k-mer).
                When the identity representation is used (:py:mod:`source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IDENTITY`),
                the k-mers just correspond to the original sequences.

            k_left (int): When gapped k-mers are used, k_left indicates the length of the k-mer left of the gap.

            k_right (int): Same as k_left, but k_right determines the length of the k-mer right of the gap

            min_gap (int): Minimum gap size when gapped k-mers are used.

            max_gap: (int): Maximum gap size when gapped k-mers are used.

        Specification:

        .. indent with spaces
        .. code-block:: yaml

                my_continuous_kmer:
                    KmerPresence:
                        sequence_encoding: CONTINUOUS_KMER
                        k: 4
                my_gapped_kmer:
                    KmerPresence:
                        sequence_encoding: GAPPED_KMER
                        k_left: 1
                        k_right: 1
                        min_gap: 1
                        max_gap: 2

        """

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_params = KmerPresenceEncoder._prepare_params(**params)
            return KmerPresenceEncoder(**prepared_params)
        else:
            raise ValueError(
                f"KmerPresenceEncoder can only be applied to repertoire dataset, got {type(dataset).__name__} instead.")

    @staticmethod
    def _prepare_params(sequence_encoding: str, normalization_type: str = "NONE", reads: str = "ALL", k: int = 0, k_left: int = 0,
                            k_right: int = 0, min_gap: int = 0, max_gap: int = 0,
                            metadata_fields_to_include: list = None, name: str = None):

        location = KmerPresenceEncoder.__name__
        ParameterValidator.assert_in_valid_list(normalization_type.upper(), ["NONE"], location, "normalization_type")
        return KmerPresenceEncoder._prepare_parameters(normalization_type, reads, sequence_encoding, k, k_left,
                            k_right, min_gap, max_gap, metadata_fields_to_include, name)

    def _encode_sequence(self, sequence: ReceptorSequence, params: EncoderParams, sequence_encoder, counts):
        params.model = vars(self)
        features = sequence_encoder.encode_sequence(sequence, params)
        if features is not None:
            for i in features:
                counts[i] = 1
        return counts