import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Union, Dict, Tuple

import bionumpy as bnp
import numpy as np

from immuneML import Constants
from immuneML.data_model.SequenceParams import RegionType
from immuneML.util.PositionHelper import PositionHelper


class KmerStrategy(ABC):
    @abstractmethod
    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None, combine_counters: bool = True) -> Union[Counter, List[Counter]]:
        """Compute k-mers for sequences
        
        Args:
            sequences: Encoded sequence array
            counts: Optional counts for each sequence
            combine_counters: If True, combine all k-mers into a single Counter, otherwise return a list of Counters
            
        Returns:
            Either a single Counter (if combine_counters=True) or a list of Counters (if combine_counters=False)
        """
        pass


class OptimizedContinuousKmerStrategy(KmerStrategy):
    def __init__(self, k: int):
        self.k = k

    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None, combine_counters: bool = True) -> Union[Counter, List[Counter]]:
        processed_sequences = sequences[sequences.lengths >= self.k]
        processed_counts = counts[sequences.lengths >= self.k] if counts is not None else None

        if processed_sequences.shape[0] < sequences.shape[0]:
            logging.warning(f"KmerFrequencyEncoder ({self.__class__.__name__}): "
                            f"Found {sequences.shape[0] - processed_sequences.shape[0]} sequences shorter than "
                            f"k={self.k}; they will be ignored.")

        if processed_sequences.shape[0] == 0:
            return Counter() if combine_counters else [Counter() for _ in range(sequences.shape[0])]
        else:
            kmers = bnp.get_kmers(processed_sequences, k=self.k)
            kmer_lists = convert_kmers_to_list(kmers, flatten=False)
            
            if combine_counters:
                if processed_counts is not None:
                    kmer_count_pairs = [(kmer, count) for kmer_list, count in zip(kmer_lists, processed_counts)
                                        for kmer in kmer_list]
                    counter = Counter({kmer: sum(count for k, count in kmer_count_pairs if k == kmer)
                                       for kmer in set(k for k, _ in kmer_count_pairs)})
                else:
                    counter = Counter([kmer for kmer_list in kmer_lists for kmer in kmer_list])
                
                if '' in counter:
                    del counter['']
                return counter
            else:
                # Create a Counter for each sequence
                counters = []
                for i in range(sequences.shape[0]):
                    if i < len(kmer_lists) and sequences.lengths[i] >= self.k:
                        if processed_counts is not None:
                            counter = Counter({kmer: processed_counts[i] for kmer in kmer_lists[i]})
                        else:
                            counter = Counter(kmer_lists[i])
                        if '' in counter:
                            del counter['']
                        counters.append(counter)
                    else:
                        counters.append(Counter())
                return counters


class OptimizedGappedKmerStrategy(KmerStrategy):
    def __init__(self, k_left: int, k_right: int, min_gap: int, max_gap: int):
        self.k_left = k_left
        self.k_right = k_right
        self.min_gap = min_gap
        self.max_gap = max_gap
        if max_gap < min_gap:
            raise ValueError(f"max_gap ({max_gap}) must be >= min_gap ({min_gap})")

    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None, combine_counters: bool = True) -> Union[Counter, List[Counter]]:
        min_length = self.k_left + self.min_gap + self.k_right
        processed_sequences = sequences[sequences.lengths >= min_length]
        processed_counts = counts[sequences.lengths >= min_length] if counts is not None else None

        if processed_sequences.shape[0] < sequences.shape[0]:
            logging.warning(f"KmerFrequencyEncoder ({self.__class__.__name__}): "
                            f"Found {sequences.shape[0] - processed_sequences.shape[0]} sequences shorter than "
                            f"minimum length {min_length}; they will be ignored.")

        if processed_sequences.shape[0] == 0:
            return Counter() if combine_counters else [Counter() for _ in range(sequences.shape[0])]

        if combine_counters:
            all_kmers = Counter()
        else:
            all_kmers = [Counter() for _ in range(sequences.shape[0])]

        for gap_size in range(self.min_gap, self.max_gap + 1):
            left_kmers = bnp.get_kmers(processed_sequences, k=self.k_left)
            shifted_seqs = processed_sequences[:, self.k_left + gap_size:]
            right_kmers = bnp.get_kmers(shifted_seqs, k=self.k_right)

            if combine_counters:
                gap_counter = self._compute_kmers_with_counts(left_kmers, right_kmers, processed_counts, gap_size) if processed_counts is not None else \
                             self._compute_kmers_without_counts(left_kmers, right_kmers, gap_size)
                all_kmers.update(gap_counter)
            else:
                # Process each sequence separately
                left_lists = convert_kmers_to_list(left_kmers, flatten=False)
                right_lists = convert_kmers_to_list(right_kmers, flatten=False)
                
                for i in range(sequences.shape[0]):
                    if i < len(left_lists) and sequences.lengths[i] >= min_length:
                        min_len = min(len(left_lists[i]), len(right_lists[i]))
                        gapped_kmers = [left_lists[i][j] + Constants.GAP_LETTER * gap_size + right_lists[i][j]
                                        for j in range(min_len)]
                        count = processed_counts[i] if processed_counts is not None else 1
                        all_kmers[i].update({kmer: count for kmer in gapped_kmers})

        return all_kmers

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


class OptimizedIMGTKmerStrategy(KmerStrategy):
    def __init__(self, k: int, region_type: RegionType = RegionType.IMGT_CDR3):
        self.k = k
        self.region_type = region_type

    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None, combine_counters: bool = True) -> Union[Counter, List[Counter]]:
        processed_sequences = sequences[sequences.lengths >= self.k]
        processed_counts = counts[sequences.lengths >= self.k] if counts is not None else None

        if processed_sequences.shape[0] < sequences.shape[0]:
            logging.warning(f"KmerFrequencyEncoder ({self.__class__.__name__}): "
                            f"Found {sequences.shape[0] - processed_sequences.shape[0]} sequences shorter than "
                            f"k={self.k}; they will be ignored.")

        if processed_sequences.shape[0] == 0:
            return Counter() if combine_counters else [Counter() for _ in range(sequences.shape[0])]

        # Group sequences by length
        unique_lengths = np.unique(processed_sequences.lengths)
        
        if combine_counters:
            all_imgt_kmers = Counter()
        else:
            all_imgt_kmers = [Counter() for _ in range(sequences.shape[0])]

        for length in unique_lengths:
            # Get sequences of current length
            length_mask = processed_sequences.lengths == length
            current_sequences = processed_sequences[length_mask]
            current_counts = processed_counts[length_mask] if processed_counts is not None else np.ones(current_sequences.shape[0], dtype=int)
            current_indices = np.where(sequences.lengths == length)[0]

            # Get IMGT positions once for this length
            positions = PositionHelper.gen_imgt_positions_from_length(length, self.region_type)
            
            if len(positions) < self.k:
                logging.warning(f"KmerFrequencyEncoder ({self.__class__.__name__}): "
                                f"Found {current_sequences.shape[0]} sequences of length {length}; "
                                f"they will be ignored.")
                continue

            # Pre-compute valid positions for k-mers
            valid_positions = positions[:-self.k+1]

            # Get k-mers for all sequences of this length at once
            kmers = bnp.get_kmers(current_sequences, k=self.k)
            kmer_lists = convert_kmers_to_list(kmers, flatten=False)
            
            if combine_counters:
                # Create all IMGT k-mers for all sequences at once
                all_kmers = []
                all_counts = []
                
                for kmers, count in zip(kmer_lists, current_counts):
                    # Create k-mers with positions for this sequence
                    imgt_kmers = [f"{kmer}-{pos}" for kmer, pos in zip(kmers, valid_positions[:len(kmers)])]
                    all_kmers.extend(imgt_kmers)
                    all_counts.extend([count] * len(imgt_kmers))
                
                # Convert to numpy arrays for faster processing
                all_kmers = np.array(all_kmers)
                all_counts = np.array(all_counts)
                
                # Get unique k-mers and sum their counts
                unique_kmers, unique_indices = np.unique(all_kmers, return_inverse=True)
                summed_counts = np.zeros(len(unique_kmers), dtype=int)
                np.add.at(summed_counts, unique_indices, all_counts)
                
                # Update counter efficiently
                all_imgt_kmers.update(dict(zip(unique_kmers, summed_counts)))
            else:
                # Process each sequence separately
                for seq_idx, (kmers, count) in enumerate(zip(kmer_lists, current_counts)):
                    imgt_kmers = [f"{kmer}-{pos}" for kmer, pos in zip(kmers, valid_positions[:len(kmers)])]
                    all_imgt_kmers[current_indices[seq_idx]].update({kmer: count for kmer in imgt_kmers})

        return all_imgt_kmers


class OptimizedIMGTGappedKmerStrategy(KmerStrategy):
    def __init__(self, k_left: int, k_right: int, min_gap: int, max_gap: int, region_type: RegionType = RegionType.IMGT_CDR3):
        self.k_left = k_left
        self.k_right = k_right
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.region_type = region_type
        if max_gap < min_gap:
            raise ValueError(f"max_gap ({max_gap}) must be >= min_gap ({min_gap})")

    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, counts: np.ndarray = None, combine_counters: bool = True) -> Union[Counter, List[Counter]]:
        min_length = self.k_left + self.min_gap + self.k_right
        processed_sequences = sequences[sequences.lengths >= min_length]
        processed_counts = counts[sequences.lengths >= min_length] if counts is not None else None

        if processed_sequences.shape[0] < sequences.shape[0]:
            logging.warning(f"KmerFrequencyEncoder ({self.__class__.__name__}): "
                            f"Found {sequences.shape[0] - processed_sequences.shape[0]} sequences shorter than "
                            f"minimum length {min_length}; they will be ignored.")

        if processed_sequences.shape[0] == 0:
            return Counter() if combine_counters else [Counter() for _ in range(sequences.shape[0])]

        # Group sequences by length
        unique_lengths = np.unique(processed_sequences.lengths)
        
        if combine_counters:
            all_imgt_kmers = Counter()
        else:
            all_imgt_kmers = [Counter() for _ in range(sequences.shape[0])]

        for length in unique_lengths:
            # Get sequences of current length
            length_mask = processed_sequences.lengths == length
            current_sequences = processed_sequences[length_mask]
            current_counts = processed_counts[length_mask] if processed_counts is not None else np.ones(current_sequences.shape[0], dtype=int)
            current_indices = np.where(sequences.lengths == length)[0]

            # Get IMGT positions once for this length
            positions = PositionHelper.gen_imgt_positions_from_length(length, self.region_type)

            if len(positions) < min_length:
                logging.warning(f"KmerFrequencyEncoder ({self.__class__.__name__}): "
                                f"Found {current_sequences.shape[0]} sequences of length {length}; "
                                f"they will be ignored.")
                continue

            # Process each gap size
            for gap_size in range(self.min_gap, self.max_gap + 1):
                # Get k-mers for all sequences of this length at once
                left_kmers = bnp.get_kmers(current_sequences, k=self.k_left)
                shifted_seqs = current_sequences[:, self.k_left + gap_size:]
                right_kmers = bnp.get_kmers(shifted_seqs, k=self.k_right)

                left_lists = convert_kmers_to_list(left_kmers, flatten=False)
                right_lists = convert_kmers_to_list(right_kmers, flatten=False)

                # Pre-compute valid positions for this gap size
                valid_positions = positions[:-min_length+1]  # Ensure we have enough positions for the full k-mer
                
                if combine_counters:
                    # Create all gapped k-mers for all sequences at once
                    all_kmers = []
                    all_counts = []
                    
                    for left_list, right_list, count in zip(left_lists, right_lists, current_counts):
                        min_len = min(len(left_list), len(right_list), len(valid_positions))
                        
                        # Create k-mers with positions for this sequence
                        if gap_size > 0:
                            # With gap
                            imgt_kmers = [
                                f"{left}.{right}-{pos}" for left, right, pos in zip(
                                    left_list[:min_len],
                                    right_list[:min_len],
                                    valid_positions[:min_len]
                                )
                            ]
                        else:
                            # Without gap (concatenated)
                            imgt_kmers = [
                                f"{left}{right}-{pos}" for left, right, pos in zip(
                                    left_list[:min_len],
                                    right_list[:min_len],
                                    valid_positions[:min_len]
                                )
                            ]
                        
                        all_kmers.extend(imgt_kmers)
                        all_counts.extend([count] * len(imgt_kmers))
                    
                    # Convert to numpy arrays for faster processing
                    all_kmers = np.array(all_kmers)
                    all_counts = np.array(all_counts)
                    
                    # Get unique k-mers and sum their counts
                    unique_kmers, unique_indices = np.unique(all_kmers, return_inverse=True)
                    summed_counts = np.zeros(len(unique_kmers), dtype=int)
                    np.add.at(summed_counts, unique_indices, all_counts)
                    
                    # Update counter efficiently
                    all_imgt_kmers.update(dict(zip(unique_kmers, summed_counts)))
                else:
                    # Process each sequence separately
                    for seq_idx, (left_list, right_list, count) in enumerate(zip(left_lists, right_lists, current_counts)):
                        min_len = min(len(left_list), len(right_list), len(valid_positions))
                        
                        if gap_size > 0:
                            imgt_kmers = [
                                f"{left}.{right}-{pos}" for left, right, pos in zip(
                                    left_list[:min_len],
                                    right_list[:min_len],
                                    valid_positions[:min_len]
                                )
                            ]
                        else:
                            imgt_kmers = [
                                f"{left}{right}-{pos}" for left, right, pos in zip(
                                    left_list[:min_len],
                                    right_list[:min_len],
                                    valid_positions[:min_len]
                                )
                            ]
                        
                        all_imgt_kmers[current_indices[seq_idx]].update({kmer: count for kmer in imgt_kmers})

        return all_imgt_kmers


class OptimizedVGeneContKmerStrategy(KmerStrategy):
    def __init__(self, k: int):
        self.k = k

    def compute_kmers(self, sequences: bnp.EncodedRaggedArray, v_genes: np.ndarray, counts: np.ndarray = None, combine_counters: bool = True) -> Union[Counter, List[Counter]]:
        processed_sequences = sequences[sequences.lengths >= self.k]
        processed_v_genes = v_genes[sequences.lengths >= self.k]
        processed_counts = counts[sequences.lengths >= self.k] if counts is not None else None

        if processed_sequences.shape[0] < sequences.shape[0]:
            logging.warning(f"KmerFrequencyEncoder ({self.__class__.__name__}): "
                            f"Found {sequences.shape[0] - processed_sequences.shape[0]} sequences shorter than "
                            f"k={self.k}; they will be ignored.")

        if processed_sequences.shape[0] == 0:
            return Counter() if combine_counters else [Counter() for _ in range(sequences.shape[0])]

        kmers = bnp.get_kmers(processed_sequences, k=self.k)
        kmer_lists = convert_kmers_to_list(kmers, flatten=False)
        
        if combine_counters:
            all_vgene_kmers = Counter()
            for seq_kmers, v_gene, count in zip(kmer_lists, processed_v_genes, 
                                              processed_counts if processed_counts is not None else [1] * len(kmer_lists)):
                for kmer in seq_kmers:
                    vgene_kmer = f"{v_gene}_{kmer}"
                    all_vgene_kmers[vgene_kmer] += count
            return all_vgene_kmers
        else:
            all_vgene_kmers = [Counter() for _ in range(sequences.shape[0])]
            for i, (seq_kmers, v_gene) in enumerate(zip(kmer_lists, processed_v_genes)):
                if sequences.lengths[i] >= self.k:
                    count = processed_counts[i] if processed_counts is not None else 1
                    for kmer in seq_kmers:
                        vgene_kmer = f"{v_gene}_{kmer}"
                        all_vgene_kmers[i][vgene_kmer] = count
            return all_vgene_kmers


def convert_kmers_to_list(kmers: bnp.EncodedArray, flatten: bool = True) -> Union[List[str], List[List[str]]]:
    if flatten:
        return [kmer for kmer_list in [kmer_str.split(",") for kmer_str in kmers.tolist()] for kmer in kmer_list]
    else:
        return [kmer_str.split(",") for kmer_str in kmers.tolist()]
