import bionumpy as bnp
import numpy as np

from immuneML.data_model.SequenceParams import RegionType
from immuneML.encodings.kmer_frequency.BNPSequenceEncodingStrategies import (
    OptimizedContinuousKmerStrategy,
    OptimizedGappedKmerStrategy,
    OptimizedIMGTKmerStrategy,
    OptimizedIMGTGappedKmerStrategy,
    OptimizedVGeneContKmerStrategy
)


def test_continuous_kmer_strategy():
    # Test setup based on KmerSequenceEncoder tests
    sequences = bnp.as_encoded_array(['CASSVFRTY'], target_encoding=bnp.AminoAcidEncoding)
    strategy = OptimizedContinuousKmerStrategy(k=3)

    # Test without counts
    result = strategy.compute_kmers(sequences)
    expected_kmers = {'CAS', 'ASS', 'SSV', 'SVF', 'VFR', 'FRT', 'RTY'}
    assert set(result.keys()) == expected_kmers
    assert all(result[kmer] == 1 for kmer in result)

    # Test with counts
    counts = np.array([2])
    result_with_counts = strategy.compute_kmers(sequences, counts)
    assert set(result_with_counts.keys()) == expected_kmers
    assert all(result_with_counts[kmer] == 2 for kmer in result_with_counts)

    # Test sequence shorter than k
    short_seq = bnp.as_encoded_array(['AC'], target_encoding=bnp.AminoAcidEncoding)
    result_short = strategy.compute_kmers(short_seq)
    assert len(result_short) == 0

    # Test empty sequences
    empty_seqs = bnp.as_encoded_array([], target_encoding=bnp.AminoAcidEncoding)
    result_empty = strategy.compute_kmers(empty_seqs)
    assert len(result_empty) == 0


def test_gapped_kmer_strategy():
    # Test setup based on GappedKmerSequenceEncoder tests
    sequences = bnp.as_encoded_array(['ACCDEFG'], target_encoding=bnp.AminoAcidEncoding)

    # Test with k_left=3, max_gap=1
    strategy1 = OptimizedGappedKmerStrategy(k_left=3, k_right=3, min_gap=0, max_gap=1)
    result1 = strategy1.compute_kmers(sequences)
    expected_kmers1 = {'ACC.EFG', 'ACCDEF', 'CCDEFG'}
    assert set(result1.keys()) == expected_kmers1

    # Test with k_left=2, k_right=3, min_gap=1
    strategy2 = OptimizedGappedKmerStrategy(k_left=2, k_right=3, min_gap=1, max_gap=1)
    result2 = strategy2.compute_kmers(sequences)
    expected_kmers2 = {'AC.DEF', 'CC.EFG'}
    assert set(result2.keys()) == expected_kmers2

    # Test with counts
    counts = np.array([2])
    result_with_counts = strategy2.compute_kmers(sequences, counts)
    assert set(result_with_counts.keys()) == expected_kmers2
    assert all(result_with_counts[kmer] == 2 for kmer in result_with_counts)

    # Test sequence shorter than minimum length
    short_seq = bnp.as_encoded_array(['ACCD'],
                                     target_encoding=bnp.AminoAcidEncoding)  # Too short for k_left=2, gap=1, k_right=3
    result_short = strategy2.compute_kmers(short_seq)
    assert len(result_short) == 0

    # Test empty sequences
    empty_seqs = bnp.as_encoded_array([], target_encoding=bnp.AminoAcidEncoding)
    result_empty = strategy2.compute_kmers(empty_seqs)
    assert len(result_empty) == 0


def test_imgt_kmer_strategy():
    # Test setup based on IMGTKmerSequenceEncoder tests
    sequences = bnp.as_encoded_array(['AHCDE'], target_encoding=bnp.AminoAcidEncoding)
    strategy = OptimizedIMGTKmerStrategy(k=3)

    # Test without counts
    result = strategy.compute_kmers(sequences)
    expected_kmers = {'AHC-105', 'HCD-106', 'CDE-107'}
    assert set(result.keys()) == expected_kmers
    assert all(result[kmer] == 1 for kmer in result)

    # Test with longer sequence
    long_seq = bnp.as_encoded_array(['CASSPRERATYEQCASSPRERATYEQCASSPRERATYEQ'], target_encoding=bnp.AminoAcidEncoding)
    result_long = strategy.compute_kmers(long_seq)
    assert {'CAS-105', 'ASS-106', 'SSP-107', 'SPR-108', 'PRE-109', 'RER-110', 'ERA-111',
            'RAT-111.1', 'ATY-111.2', 'TYE-111.3', 'YEQ-111.4', 'EQC-111.5',
            'QCA-111.6', 'CAS-111.7', 'ASS-111.8', 'SSP-111.9', 'SPR-111.10',
            'PRE-111.11', 'RER-111.12', 'ERA-111.13', 'RAT-112.13', 'ATY-112.12',
            'TYE-112.11', 'YEQ-112.10', 'EQC-112.9', 'QCA-112.8', 'CAS-112.7',
            'ASS-112.6', 'SSP-112.5', 'SPR-112.4', 'PRE-112.3', 'RER-112.2',
            'ERA-112.1', 'RAT-112', 'ATY-113', 'TYE-114', 'YEQ-115'} == set(result_long)

    # Test with counts
    counts = np.array([2])
    result_with_counts = strategy.compute_kmers(sequences, counts)
    assert set(result_with_counts.keys()) == expected_kmers
    assert all(result_with_counts[kmer] == 2 for kmer in result_with_counts)

    # Test sequence shorter than k
    short_seq = bnp.as_encoded_array(['AC'], target_encoding=bnp.AminoAcidEncoding)
    result_short = strategy.compute_kmers(short_seq)
    assert len(result_short) == 0

    # Test empty sequences
    empty_seqs = bnp.as_encoded_array([], target_encoding=bnp.AminoAcidEncoding)
    result_empty = strategy.compute_kmers(empty_seqs)
    assert len(result_empty) == 0

    # Test with different region type
    junction_seq = bnp.as_encoded_array(['CASSVDRTYEQ'], target_encoding=bnp.AminoAcidEncoding)
    junction_strategy = OptimizedIMGTKmerStrategy(k=3, region_type=RegionType.IMGT_JUNCTION)
    result_junction = junction_strategy.compute_kmers(junction_seq)
    assert 'CAS-104' in result_junction  # Junction should start at position 104


def test_vgene_kmer_strategy():
    # Test setup based on VGeneContKmerEncoder tests
    sequences = bnp.as_encoded_array(['CASSVFRTY'], target_encoding=bnp.AminoAcidEncoding)
    v_genes = np.array(['TRBV1'])
    strategy = OptimizedVGeneContKmerStrategy(k=3)

    # Test without counts
    result = strategy.compute_kmers(sequences, v_genes)
    expected_kmers = {'TRBV1_CAS', 'TRBV1_ASS', 'TRBV1_SSV', 'TRBV1_SVF', 'TRBV1_VFR', 'TRBV1_FRT', 'TRBV1_RTY'}
    assert set(result.keys()) == expected_kmers
    assert all(result[kmer] == 1 for kmer in result)

    # Test with counts
    counts = np.array([2])
    result_with_counts = strategy.compute_kmers(sequences, v_genes, counts)
    assert set(result_with_counts.keys()) == expected_kmers
    assert all(result_with_counts[kmer] == 2 for kmer in result_with_counts)

    # Test with multiple sequences
    multi_seqs = bnp.as_encoded_array(['CASSVFRTY', 'CASSACC'], target_encoding=bnp.AminoAcidEncoding)
    multi_v_genes = np.array(['TRBV1', 'TRBV2'])
    result_multi = strategy.compute_kmers(multi_seqs, multi_v_genes)
    assert 'TRBV1_CAS' in result_multi
    assert 'TRBV2_CAS' in result_multi

    # Test sequence shorter than k
    short_seq = bnp.as_encoded_array(['AC'], target_encoding=bnp.AminoAcidEncoding)
    short_v_genes = np.array(['TRBV1'])
    result_short = strategy.compute_kmers(short_seq, short_v_genes)
    assert len(result_short) == 0

    # Test empty sequences
    empty_seqs = bnp.as_encoded_array([], target_encoding=bnp.AminoAcidEncoding)
    empty_v_genes = np.array([])
    result_empty = strategy.compute_kmers(empty_seqs, empty_v_genes)
    assert len(result_empty) == 0


def test_imgt_gapped_kmer_strategy():
    # Test setup based on IMGTGappedKmerEncoder tests
    sequences = bnp.as_encoded_array(['AHCDE'], target_encoding=bnp.AminoAcidEncoding)
    strategy = OptimizedIMGTGappedKmerStrategy(k_left=1, k_right=1, min_gap=0, max_gap=1)

    # Test without counts
    result = strategy.compute_kmers(sequences)
    expected_kmers = {'AH-105', 'HC-106', 'CD-107', 'DE-116', 'A.C-105', 'H.D-106', 'C.E-107'}
    assert set(result.keys()) == expected_kmers
    assert all(result[kmer] == 1 for kmer in result)

    # Test with longer sequence
    long_seq = bnp.as_encoded_array(['CASSPRERATYEQCAY'], target_encoding=bnp.AminoAcidEncoding)
    result_long = strategy.compute_kmers(long_seq)
    expected_long_kmers = {'CA-105', 'AS-106', 'SS-107', 'SP-108', 'PR-109', 'RE-110', 'ER-111',
                           'RA-111.1', 'AT-112.2', 'TY-112.1', 'YE-112', 'EQ-113', 'QC-114',
                           'CA-115', 'AY-116', 'C.S-105', 'A.S-106', 'S.P-107', 'S.R-108', 'P.E-109',
                           'R.R-110', 'E.A-111', 'R.T-111.1', 'A.Y-112.2', 'T.E-112.1', 'Y.Q-112',
                           'E.C-113', 'Q.A-114', 'C.Y-115'}
    assert set(result_long.keys()) == expected_long_kmers

    # Test with counts
    counts = np.array([2])
    result_with_counts = strategy.compute_kmers(sequences, counts)
    assert set(result_with_counts.keys()) == expected_kmers
    assert all(result_with_counts[kmer] == 2 for kmer in result_with_counts)

    # Test sequence shorter than minimum length
    short_seq = bnp.as_encoded_array(['A'], target_encoding=bnp.AminoAcidEncoding)
    result_short = strategy.compute_kmers(short_seq)
    assert len(result_short) == 0

    # Test empty sequences
    empty_seqs = bnp.as_encoded_array([], target_encoding=bnp.AminoAcidEncoding)
    result_empty = strategy.compute_kmers(empty_seqs)
    assert len(result_empty) == 0

    # Test with different region type
    junction_seq = bnp.as_encoded_array(['CASSVDRTYEQ'], target_encoding=bnp.AminoAcidEncoding)
    junction_strategy = OptimizedIMGTGappedKmerStrategy(k_left=1, k_right=1, min_gap=0, max_gap=1,
                                                        region_type=RegionType.IMGT_JUNCTION)
    result_junction = junction_strategy.compute_kmers(junction_seq)
    assert 'C.S-104' in result_junction  # Junction should start at position 104
