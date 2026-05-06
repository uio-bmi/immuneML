"""
Vectorised k-mer extraction using bionumpy EncodedRaggedArray batches.

All six SequenceEncodingType variants are covered:
  CONTINUOUS_KMER      – bnp.get_kmers + label lookup
  GAPPED_KMER          – raw-code index arrays for left/right windows
  IMGT_CONTINUOUS_KMER – bnp.get_kmers + per-unique-length IMGT position tables
  IMGT_GAPPED_KMER     – gapped raw-code approach + IMGT position annotation
  V_GENE_CONT_KMER     – continuous kmers with v_gene prefix
  V_GENE_IMGT_KMER     – IMGT continuous kmers with v_gene prefix

Each public encode_* function returns:
    flat_kmers : np.ndarray[str]  – all kmer strings concatenated across sequences
    row_ids    : np.ndarray[int]  – row_ids[m] = source sequence index for kmer m
                                    (sequences too short contribute zero kmers)
"""

import logging
from typing import Dict, List, Optional, Tuple

import bionumpy as bnp
import numpy as np
from bionumpy.encoded_array import EncodedRaggedArray

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import get_sequence_field_name
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PositionHelper import PositionHelper
from immuneML.util.ReadsType import ReadsType

# ── type aliases ─────────────────────────────────────────────────────────────
_KmerResult = Tuple[np.ndarray, np.ndarray]   # (flat_kmers, row_ids)

V_GENE_ENCODING_TYPES = frozenset({
    SequenceEncodingType.V_GENE_CONT_KMER,
    SequenceEncodingType.V_GENE_IMGT_KMER,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _alphabet_array(seq_array: EncodedRaggedArray) -> np.ndarray:
    alphabet_str = bytes(seq_array.encoding._alphabet).decode("ascii")
    return np.array(list(alphabet_str))


def _kmer_label_array(kmers) -> np.ndarray:
    return np.array(kmers.encoding.get_labels())


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """[0..n0-1, 0..n1-1, …] without a Python loop over groups."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.intp)
    cum = np.concatenate([[0], np.cumsum(counts[:-1])])
    rep_ids = np.repeat(np.arange(len(counts)), counts)
    return np.arange(total) - cum[rep_ids]


def _to_row_ids(valid_mask: np.ndarray, valid_kmer_lens: np.ndarray) -> np.ndarray:
    """Map from (valid-sequence index, kmer position) to original sequence index."""
    valid_indices = np.where(valid_mask)[0]
    rep_ids = np.repeat(np.arange(len(valid_kmer_lens)), valid_kmer_lens)
    return valid_indices[rep_ids]


def _filter_valid(lengths: np.ndarray, min_len: int, encoding_name: str) -> np.ndarray:
    valid_mask = lengths >= min_len
    if not valid_mask.all():
        logging.warning(f"{encoding_name}: {(~valid_mask).sum()} sequence(s) "
                        f"shorter than {min_len}, ignored.")
    return valid_mask


def _join_char_columns(char_matrix: np.ndarray) -> np.ndarray:
    strs = char_matrix[:, 0].copy()
    for col in range(1, char_matrix.shape[1]):
        strs = np.char.add(strs, char_matrix[:, col])
    return strs


# ── prefix helpers ────────────────────────────────────────────────────────────

def _apply_prefix(flat_str: np.ndarray, per_seq_labels, row_ids: np.ndarray) -> np.ndarray:
    per_kmer = np.asarray(per_seq_labels)[row_ids]
    return np.char.add(np.char.add(per_kmer, Constants.FEATURE_DELIMITER), flat_str)


def _apply_locus_prefix(flat_str: np.ndarray, locus_labels: Optional[List[str]],
                        row_ids: np.ndarray) -> np.ndarray:
    if locus_labels is None:
        return flat_str
    return _apply_prefix(flat_str, locus_labels, row_ids)


def _apply_v_gene_prefix(flat_str: np.ndarray, v_genes: List[str],
                          row_ids: np.ndarray) -> np.ndarray:
    return _apply_prefix(flat_str, v_genes, row_ids)


# ── IMGT position helpers ─────────────────────────────────────────────────────

def _imgt_positions_cache(valid_lengths: np.ndarray,
                           region_type: RegionType) -> Dict[int, Optional[list]]:
    return {int(ul): PositionHelper.gen_imgt_positions_from_length(int(ul), region_type)
            for ul in np.unique(valid_lengths)}


def _lookup_imgt_pos_array(valid_lengths: np.ndarray, kmer_pos: np.ndarray,
                            rep_ids: np.ndarray, region_type: RegionType) -> np.ndarray:
    """Return IMGT position string for each kmer (one entry per kmer in the batch)."""
    pos_cache = _imgt_positions_cache(valid_lengths, region_type)
    result = np.full(len(kmer_pos), "", dtype="<U10")
    for length, positions in pos_cache.items():
        if not positions:
            continue
        mask = valid_lengths[rep_ids] == length
        result[mask] = np.asarray(positions, dtype=str)[kmer_pos[mask]]
    return result


def _annotate_imgt(flat_str: np.ndarray, valid_lengths: np.ndarray,
                   kmer_pos: np.ndarray, rep_ids: np.ndarray,
                   region_type: RegionType) -> np.ndarray:
    imgt_pos = _lookup_imgt_pos_array(valid_lengths, kmer_pos, rep_ids, region_type)
    return np.char.add(np.char.add(flat_str, Constants.FEATURE_DELIMITER), imgt_pos)


# ── gapped kmer helpers ───────────────────────────────────────────────────────

def _build_gapped_strings(flat_raw: np.ndarray, seq_offsets: np.ndarray,
                           k_left: int, k_right: int, g: int,
                           alphabet: np.ndarray) -> np.ndarray:
    left_pos = seq_offsets[:, np.newaxis] + np.arange(k_left)
    right_pos = seq_offsets[:, np.newaxis] + (k_left + g) + np.arange(k_right)
    left_strs = _join_char_columns(alphabet[flat_raw[left_pos]])
    right_strs = _join_char_columns(alphabet[flat_raw[right_pos]])
    return np.char.add(np.char.add(left_strs, "." * g), right_strs)


def _extract_gap(flat_raw: np.ndarray, lengths: np.ndarray, all_starts: np.ndarray,
                  alphabet: np.ndarray, k_left: int, k_right: int,
                  g: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract gapped kmers for a single gap size g.

    Returns (kmer_strs, valid_lengths, kmer_pos, rep_ids_original) or None if no
    valid sequences exist for this gap size.
    """
    window = k_left + g + k_right
    valid_mask = lengths >= window
    if not valid_mask.any():
        return None

    valid_lengths = lengths[valid_mask]
    valid_starts = all_starts[valid_mask]
    n_kmers_per_seq = valid_lengths - window + 1

    kmer_pos = _ragged_arange(n_kmers_per_seq)
    rep_ids = np.repeat(np.arange(len(n_kmers_per_seq)), n_kmers_per_seq)
    seq_offsets = valid_starts[rep_ids] + kmer_pos
    kmer_strs = _build_gapped_strings(flat_raw, seq_offsets, k_left, k_right, g, alphabet)

    original_row_ids = np.where(valid_mask)[0][rep_ids]
    return kmer_strs, valid_lengths, kmer_pos, rep_ids, original_row_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Private extraction cores (no locus/v_gene prefix)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_continuous(seq_array: EncodedRaggedArray, k: int) -> _KmerResult:
    lengths = np.asarray(seq_array.lengths)
    valid_mask = _filter_valid(lengths, k, "CONTINUOUS_KMER")
    if not valid_mask.any():
        return np.empty(0, dtype=str), np.empty(0, dtype=np.intp)

    kmers = bnp.get_kmers(seq_array[valid_mask], k)
    labels = _kmer_label_array(kmers)
    flat_str = labels[kmers.ravel().raw()]
    valid_kmer_lens = np.asarray(kmers.lengths)
    return flat_str, _to_row_ids(valid_mask, valid_kmer_lens)


def _extract_gapped(seq_array: EncodedRaggedArray,
                     k_left: int, k_right: int,
                     min_gap: int, max_gap: int) -> _KmerResult:
    lengths = np.asarray(seq_array.lengths)
    flat_raw = seq_array.ravel().raw()
    all_starts = np.concatenate([[0], np.cumsum(lengths[:-1])])
    alphabet = _alphabet_array(seq_array)

    all_kmers, all_row_ids = [], []
    for g in range(min_gap, max_gap + 1):
        result = _extract_gap(flat_raw, lengths, all_starts, alphabet, k_left, k_right, g)
        if result is not None:
            kmer_strs, _, _, _, original_row_ids = result
            all_kmers.append(kmer_strs)
            all_row_ids.append(original_row_ids)

    if not all_kmers:
        return np.empty(0, dtype=str), np.empty(0, dtype=np.intp)
    return np.concatenate(all_kmers), np.concatenate(all_row_ids)


def _extract_imgt(seq_array: EncodedRaggedArray, k: int,
                   region_type: RegionType) -> _KmerResult:
    lengths = np.asarray(seq_array.lengths)
    valid_mask = _filter_valid(lengths, k, "IMGT_CONTINUOUS_KMER")
    if not valid_mask.any():
        return np.empty(0, dtype=str), np.empty(0, dtype=np.intp)

    valid_lengths = lengths[valid_mask]
    kmers = bnp.get_kmers(seq_array[valid_mask], k)
    labels = _kmer_label_array(kmers)
    flat_str = labels[kmers.ravel().raw()]

    valid_kmer_lens = np.asarray(kmers.lengths)
    kmer_pos = _ragged_arange(valid_kmer_lens)
    rep_ids = np.repeat(np.arange(len(valid_kmer_lens)), valid_kmer_lens)

    flat_str = _annotate_imgt(flat_str, valid_lengths, kmer_pos, rep_ids, region_type)
    return flat_str, _to_row_ids(valid_mask, valid_kmer_lens)


def _extract_imgt_gapped(seq_array: EncodedRaggedArray,
                          k_left: int, k_right: int,
                          min_gap: int, max_gap: int,
                          region_type: RegionType) -> _KmerResult:
    lengths = np.asarray(seq_array.lengths)
    flat_raw = seq_array.ravel().raw()
    all_starts = np.concatenate([[0], np.cumsum(lengths[:-1])])
    alphabet = _alphabet_array(seq_array)

    all_kmers, all_row_ids = [], []
    for g in range(min_gap, max_gap + 1):
        result = _extract_gap(flat_raw, lengths, all_starts, alphabet, k_left, k_right, g)
        if result is None:
            continue
        kmer_strs, valid_lengths_g, kmer_pos, rep_ids, original_row_ids = result
        kmer_strs = _annotate_imgt(kmer_strs, valid_lengths_g, kmer_pos, rep_ids, region_type)
        all_kmers.append(kmer_strs)
        all_row_ids.append(original_row_ids)

    if not all_kmers:
        return np.empty(0, dtype=str), np.empty(0, dtype=np.intp)
    return np.concatenate(all_kmers), np.concatenate(all_row_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def encode_continuous_kmer(seq_array: EncodedRaggedArray, k: int,
                            locus_labels: Optional[List[str]] = None) -> _KmerResult:
    flat_str, row_ids = _extract_continuous(seq_array, k)
    return _apply_locus_prefix(flat_str, locus_labels, row_ids), row_ids


def encode_gapped_kmer(seq_array: EncodedRaggedArray,
                        k_left: int, k_right: int, min_gap: int, max_gap: int,
                        locus_labels: Optional[List[str]] = None) -> _KmerResult:
    flat_str, row_ids = _extract_gapped(seq_array, k_left, k_right, min_gap, max_gap)
    return _apply_locus_prefix(flat_str, locus_labels, row_ids), row_ids


def encode_imgt_kmer(seq_array: EncodedRaggedArray, k: int, region_type: RegionType,
                      locus_labels: Optional[List[str]] = None) -> _KmerResult:
    flat_str, row_ids = _extract_imgt(seq_array, k, region_type)
    return _apply_locus_prefix(flat_str, locus_labels, row_ids), row_ids


def encode_imgt_gapped_kmer(seq_array: EncodedRaggedArray,
                              k_left: int, k_right: int, min_gap: int, max_gap: int,
                              region_type: RegionType,
                              locus_labels: Optional[List[str]] = None) -> _KmerResult:
    flat_str, row_ids = _extract_imgt_gapped(seq_array, k_left, k_right, min_gap, max_gap, region_type)
    return _apply_locus_prefix(flat_str, locus_labels, row_ids), row_ids


def encode_v_gene_kmer(seq_array: EncodedRaggedArray, k: int, v_genes: List[str],
                        locus_labels: Optional[List[str]] = None) -> _KmerResult:
    flat_str, row_ids = _extract_continuous(seq_array, k)
    flat_str = _apply_v_gene_prefix(flat_str, v_genes, row_ids)
    return _apply_locus_prefix(flat_str, locus_labels, row_ids), row_ids


def encode_v_gene_imgt_kmer(seq_array: EncodedRaggedArray, k: int, region_type: RegionType,
                              v_genes: List[str],
                              locus_labels: Optional[List[str]] = None) -> _KmerResult:
    flat_str, row_ids = _extract_imgt(seq_array, k, region_type)
    flat_str = _apply_v_gene_prefix(flat_str, v_genes, row_ids)
    return _apply_locus_prefix(flat_str, locus_labels, row_ids), row_ids


def seq_field(region_type: RegionType, sequence_type: SequenceType) -> str:
    if region_type == RegionType.FULL_SEQUENCE:
        return "sequence_aa" if sequence_type == SequenceType.AMINO_ACID else "sequence"
    return get_sequence_field_name(region_type, sequence_type)


def get_v_genes(data, mask: Optional[np.ndarray] = None) -> List[str]:
    v_calls = np.asarray(data.v_call.tolist())
    if mask is not None:
        v_calls = v_calls[mask]
    return [vc.split("*")[0] if vc else "" for vc in v_calls]


def kmer_weights(data, reads: ReadsType, row_ids: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if reads != ReadsType.ALL or len(row_ids) == 0:
        return None
    dup_counts = np.asarray(data.duplicate_count.tolist(), dtype=float)
    if mask is not None:
        dup_counts = dup_counts[mask]
    return dup_counts[row_ids]


def dispatch_encoding(seq_array: EncodedRaggedArray, encoding_type: SequenceEncodingType,
                       k: int, k_left: int, k_right: int, min_gap: int, max_gap: int,
                       region_type: RegionType,
                       v_genes: Optional[List[str]] = None,
                       locus_labels: Optional[List[str]] = None) -> _KmerResult:
    if encoding_type == SequenceEncodingType.CONTINUOUS_KMER:
        return encode_continuous_kmer(seq_array, k, locus_labels)
    if encoding_type == SequenceEncodingType.GAPPED_KMER:
        return encode_gapped_kmer(seq_array, k_left, k_right, min_gap, max_gap, locus_labels)
    if encoding_type == SequenceEncodingType.IMGT_CONTINUOUS_KMER:
        return encode_imgt_kmer(seq_array, k, region_type, locus_labels)
    if encoding_type == SequenceEncodingType.IMGT_GAPPED_KMER:
        return encode_imgt_gapped_kmer(seq_array, k_left, k_right, min_gap, max_gap, region_type, locus_labels)
    if encoding_type == SequenceEncodingType.V_GENE_CONT_KMER:
        return encode_v_gene_kmer(seq_array, k, v_genes, locus_labels)
    if encoding_type == SequenceEncodingType.V_GENE_IMGT_KMER:
        return encode_v_gene_imgt_kmer(seq_array, k, region_type, v_genes, locus_labels)
    raise ValueError(f"Unknown SequenceEncodingType: {encoding_type}")