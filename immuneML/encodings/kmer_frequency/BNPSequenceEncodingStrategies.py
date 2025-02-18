from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Union

import bionumpy as bnp
import numpy as np

from immuneML import Constants


class KmerStrategy(ABC):
    @abstractmethod
    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None) -> Counter:
        pass


class OptimizedContinuousKmerStrategy(KmerStrategy):
    def __init__(self, k: int):
        self.k = k

    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None) -> Counter:
        kmers = bnp.get_kmers(sequences, k=self.k)

        if counts is not None:
            kmer_count_pairs = [(kmer, count) for kmer_list, count in zip(kmers.tolist(), counts)
                                for kmer in kmer_list.split(",")]
            counter = Counter({kmer: sum(count for k, count in kmer_count_pairs if k == kmer)
                               for kmer in set(k for k, _ in kmer_count_pairs)})
        else:
            kmers = convert_kmers_to_list(kmers, flatten=True)
            counter = Counter(kmers)

        if '' in counter.keys():
            del counter['']

        return counter


class OptimizedGappedKmerStrategy(KmerStrategy):
    def __init__(self, k_left: int, k_right: int, min_gap: int, max_gap: int):
        self.k_left = k_left
        self.k_right = k_right
        self.min_gap = min_gap
        self.max_gap = max_gap
        if max_gap < min_gap:
            raise ValueError(f"max_gap ({max_gap}) must be >= min_gap ({min_gap})")

    def _compute_kmers_with_counts(self, left_kmers, right_kmers, counts, gap_size):
        left_lists = convert_kmers_to_list(left_kmers, flatten=False)
        right_lists = convert_kmers_to_list(right_kmers, flatten=False)

        kmer_count_pairs = []
        for left_list, right_list, count in zip(left_lists, right_lists, counts):
            min_len = min(len(left_list), len(right_list))
            gapped_kmers = [left_list[i] + Constants.GAP_LETTER * gap_size + right_list[i]
                            for i in range(min_len)]
            kmer_count_pairs.extend((kmer, count) for kmer in gapped_kmers)

        gap_counter = Counter({kmer: sum(count for k, count in kmer_count_pairs if k == kmer)
                               for kmer in set(k for k, _ in kmer_count_pairs)})

        return gap_counter


    def _compute_kmers_without_counts(self, left_kmers, right_kmers, gap_size):
        left_lists = convert_kmers_to_list(left_kmers, flatten=False)
        right_lists = convert_kmers_to_list(right_kmers, flatten=False)

        all_gapped_kmers = []
        for left_list, right_list in zip(left_lists, right_lists):
            min_len = min(len(left_list), len(right_list))
            gapped_kmers = [left_list[i] + Constants.GAP_LETTER * gap_size + right_list[i]
                            for i in range(min_len)]
            all_gapped_kmers.extend(gapped_kmers)

        gap_counter = Counter(all_gapped_kmers)
        return gap_counter

    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None) -> Counter:
        all_kmers = Counter()

        for gap_size in range(self.min_gap, self.max_gap + 1):

            left_kmers = bnp.get_kmers(sequences, k=self.k_left)

            shifted_seqs = sequences[:, self.k_left + gap_size:]
            right_kmers = bnp.get_kmers(shifted_seqs, k=self.k_right)

            if counts is not None:
                gap_counter = self._compute_kmers_with_counts(left_kmers, right_kmers, counts, gap_size)
            else:
                gap_counter = self._compute_kmers_without_counts(left_kmers, right_kmers, gap_size)

            all_kmers.update(gap_counter)

        return all_kmers


def convert_kmers_to_list(kmers: bnp.EncodedArray, flatten: bool = True) -> Union[List[str], List[List[str]]]:
    if flatten:
        return [kmer for kmer_list in [kmer_str.split(",") for kmer_str in kmers.tolist()] for kmer in kmer_list]
    else:
        return [kmer_str.split(",") for kmer_str in kmers.tolist()]
