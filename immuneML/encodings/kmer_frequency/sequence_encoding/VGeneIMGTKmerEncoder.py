import logging

from immuneML import Constants
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from immuneML.util.KmerHelper import KmerHelper


class VGeneIMGTKmerEncoder(SequenceEncodingStrategy):

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
        sequence_type = params.model.get('sequence_type', params.sequence_type)

        length = len(sequence.get_sequence(sequence_type))

        if length < k:
            logging.warning(f'KmerSequenceEncoder: Sequence length {length} is less than {k}. Ignoring sequence...')
            return None

        kmers = KmerHelper.create_IMGT_kmers_from_sequence(sequence=sequence, k=k, sequence_type=sequence_type,
                                                           region_type=params.model.get('region_type',
                                                                                        params.region_type))

        kmers = [Constants.FEATURE_DELIMITER.join([sequence.v_gene] + [str(mer) for mer in kmer]) for kmer in kmers]

        return kmers

    @staticmethod
    def get_feature_names(params: EncoderParams):
        return ["sequence"]
