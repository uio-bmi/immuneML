import bionumpy as bnp
import numpy as np

from immuneML import Constants
from immuneML.encodings.kmer_frequency.BNPSequenceEncodingStrategies import (
    OptimizedContinuousKmerStrategy,
    OptimizedGappedKmerStrategy
)


def create_sequence_array(sequences):
    return bnp.as_encoded_array(sequences, target_encoding=bnp.DNAEncoding)


def test_continuous_kmer_strategy():
    # Test setup
    sequences = bnp.as_encoded_array(['ACGT', 'ACGT', 'TGCA'], target_encoding=bnp.DNAEncoding)
    counts = np.array([2, 1, 3])
    strategy = OptimizedContinuousKmerStrategy(k=2)

    # Test without counts
    result = strategy.compute_kmers(sequences)
    assert result['AC'] == 2
    assert result['CG'] == 2
    assert result['GT'] == 2
    assert result['TG'] == 1
    assert result['GC'] == 1
    assert result['CA'] == 1

    # Test with counts
    result_with_counts = strategy.compute_kmers(sequences, counts)
    assert result_with_counts['AC'] == 3
    assert result_with_counts['TG'] == 3
    assert result_with_counts['CA'] == 3


def test_gapped_kmer_strategy():
    # Test setup
    sequences = bnp.as_encoded_array(['ACGTACGTA', 'TGCATGCA'], target_encoding=bnp.DNAEncoding)
    counts = np.array([2, 3])
    strategy = OptimizedGappedKmerStrategy(k_left=2, k_right=2, min_gap=0, max_gap=1)

    # Test without counts
    result = strategy.compute_kmers(sequences)
    print(result)
    
    assert result['ACGT'] == 2
    assert result['TGCA'] == 2
    
    gapped_kmer = f"AC{Constants.GAP_LETTER}TA"
    assert result[gapped_kmer] == 2
    
    result_with_counts = strategy.compute_kmers(sequences, counts)
    assert result_with_counts['ACGT'] == 4
    assert result_with_counts['TGCA'] == 6
    
    gapped_kmer = f"AC{Constants.GAP_LETTER}TA"
    assert result_with_counts[gapped_kmer] == 4


def test_edge_cases():
    # Test empty sequences
    sequences = create_sequence_array([])
    strategy = OptimizedContinuousKmerStrategy(k=2)
    result = strategy.compute_kmers(sequences)
    assert len(result) == 0
    
    # Test sequences shorter than k
    sequences = create_sequence_array(['A', 'AC'])
    strategy = OptimizedContinuousKmerStrategy(k=3)
    result = strategy.compute_kmers(sequences)
    assert len(result) == 0
