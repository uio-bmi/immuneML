# quality: peripheral
import itertools
import logging
import warnings

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PositionHelper import PositionHelper


class KmerHelper:
    @staticmethod
    def create_kmers_from_sequence(sequence: ReceptorSequence, k: int, sequence_type: SequenceType, overlap: bool = True):
        return KmerHelper.create_kmers_from_string(sequence.get_sequence(sequence_type), k, overlap)

    @staticmethod
    def create_kmers_from_string(sequence, k: int, overlap: bool = True):
        kmers = []
        step = 1 if overlap else k
        for i in range(0, len(sequence) - k + 1, step):
            kmers.append(sequence[i:i + k])
        return kmers

    @staticmethod
    def create_IMGT_kmers_from_sequence(sequence: ReceptorSequence, k: int, sequence_type: SequenceType):
        return KmerHelper.create_IMGT_kmers_from_string(sequence.get_sequence(sequence_type), k, sequence.get_attribute("region_type"))

    @staticmethod
    def create_IMGT_kmers_from_string(sequence: str, k: int, region_type: RegionType):
        positions = PositionHelper.gen_imgt_positions_from_length(len(sequence), region_type)
        if positions is not None and len(positions) > 0:
            sequence_w_pos = list(zip(list(sequence), positions))
            kmers = KmerHelper.create_kmers_from_string(sequence_w_pos, k)
            kmers = [(''.join([x[0] for x in kmer]), kmer[0][1]) for kmer in kmers]
            return kmers
        else:
            logging.warning(f"{KmerHelper.__name__}: {sequence} could not be represented using IMGT {k}-mers, "
                            f"no IMGT positions were found. Returning empty list instead...")
            return []

    @staticmethod
    def create_IMGT_gapped_kmers_from_sequence(sequence: ReceptorSequence, sequence_type: SequenceType, k_left: int, max_gap: int, k_right: int = None, min_gap: int = 0):
        positions = PositionHelper.gen_imgt_positions_from_sequence(sequence, sequence_type)

        sequence_w_pos = list(zip(list(sequence.get_sequence(sequence_type)), positions))
        kmers = KmerHelper.create_gapped_kmers_from_string(sequence_w_pos, k_left=k_left, max_gap=max_gap,
                                                           k_right=k_right, min_gap=min_gap)
        if kmers is not None:
            kmers = [(''.join([x[0] for x in kmer]), kmer[0][1]) for kmer in kmers]
            return kmers
        else:
            return None

    @staticmethod
    def create_gapped_kmers_from_string(sequence, k_left: int, max_gap: int, k_right: int = None, min_gap: int = 0):
        length = len(sequence)
        k_right = k_left if k_right is None else k_right

        if length < k_left + k_right + max_gap:
            raise ValueError('Sequence length is less than k_left + k_right + max_gap. '
                             'Filter sequences from each repertoire that are less than this length then rerun.')
        gapped_kmers = []
        for i in range(min_gap, max_gap + 1):
            s = k_left + k_right + i
            kmers = [sequence[i: i + s] for i in range(length - s + 1)]
            if isinstance(sequence, str):
                gapped_kmers.extend([kmer[:k_left] + i * "." + kmer[k_left + i:] for kmer in kmers])
            if isinstance(sequence, list):
                gapped_kmers.extend([kmer[:k_left] + [(".", el[1]) for el in kmer[k_left:k_left+i]] + kmer[k_left + i:] for kmer in kmers])
        return gapped_kmers

    @staticmethod
    def create_gapped_kmers_from_sequence(sequence: ReceptorSequence, sequence_type: SequenceType, k_left: int, max_gap: int, k_right: int = None,
                                          min_gap: int = 0):
        return KmerHelper.create_gapped_kmers_from_string(sequence.get_sequence(sequence_type), k_left, max_gap, k_right, min_gap)

    @staticmethod
    def create_all_kmers(k: int, alphabet: list):
        """
        creates all possible k-mers given a k-mer length and an alphabet
        :param k: length of k-mer (int)
        :param alphabet: list of characters from which to make all possible k-mers (list)
        :return: alphabetically sorted list of k-mers
        """
        kmers = [''.join(x) for x in itertools.product(alphabet, repeat=k)]
        kmers.sort()
        return kmers

    @staticmethod
    def create_sentences_from_repertoire(repertoire: Repertoire, k: int, sequence_type: SequenceType, overlap: bool = True):
        sentences = []
        for sequence in repertoire.sequences:
            sentences.append(KmerHelper.create_kmers_from_sequence(sequence=sequence, k=k, overlap=overlap, sequence_type=sequence_type))
        return sentences

    @staticmethod
    def create_kmers_within_HD(kmer: str, alphabet: list, distance: int = 1):

        assert distance < len(kmer)

        if distance > 1:
            warnings.warn("In create_kmers_within_HD distance larger than 1 is not yet implemented. "
                          "Using default value 1...", Warning)

        pairs = []

        for i in range(len(kmer)):
            for letter in alphabet:
                new_kmer = kmer[0:i] + letter + kmer[i + 1:]
                pairs.append([kmer, new_kmer])

        return pairs
