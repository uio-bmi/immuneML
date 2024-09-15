import numpy as np
from bionumpy import DNAEncoding, AminoAcidEncoding
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy.encodings import AlphabetEncoding
import bionumpy as bnp


@bnpdataclass
class AIRRSequenceSet:
    sequence_id: str = None
    sequence: DNAEncoding = None
    quality: str = None
    sequence_aa: AminoAcidEncoding = None
    rev_comp: bool = None
    productive: bool = None
    vj_in_frame: bool = None
    stop_codon: bool = None
    complete_vdj: bool = None
    locus: str = None
    locus_species: str = None
    v_call: str = None
    d_call: str = None
    d2_call: str = None
    j_call: str = None
    c_call: str = None
    sequence_alignment: str = None
    quality_alignment: str = None
    sequence_alignment_aa: str = None
    germline_alignment: str = None
    germline_alignment_aa: str = None
    junction: DNAEncoding = None
    junction_aa: AminoAcidEncoding = None
    np1: DNAEncoding = None
    np1_aa: AminoAcidEncoding = None
    np2: DNAEncoding = None
    np2_aa: AminoAcidEncoding = None
    np3: DNAEncoding = None
    np3_aa: AminoAcidEncoding = None
    cdr1: DNAEncoding = None
    cdr1_aa: AminoAcidEncoding = None
    cdr2: DNAEncoding = None
    cdr2_aa: AminoAcidEncoding = None
    cdr3: DNAEncoding = None
    cdr3_aa: AminoAcidEncoding = None
    fwr1: DNAEncoding = None
    fwr1_aa: AminoAcidEncoding = None
    fwr2: DNAEncoding = None
    fwr2_aa: AminoAcidEncoding = None
    fwr3: DNAEncoding = None
    fwr3_aa: AminoAcidEncoding = None
    fwr4: DNAEncoding = None
    fwr4_aa: AminoAcidEncoding = None
    v_score: float = None
    v_identity: float = None
    v_support: float = None
    v_cigar: str = None
    d_score: float = None
    d_identity: float = None
    d_support: float = None
    d_cigar: str = None
    d2_score: float = None
    d2_identity: float = None
    d2_support: float = None
    d2_cigar: str = None
    j_score: float = None
    j_identity: float = None
    j_support: float = None
    j_cigar: str = None
    c_score: float = None
    c_identity: float = None
    c_support: float = None
    c_cigar: str = None
    v_sequence_start: int = None
    v_sequence_end: int = None
    v_germline_start: int = None
    v_germline_end: int = None
    v_alignment_start: int = None
    v_alignment_end: int = None
    d_sequence_start: int = None
    d_sequence_end: int = None
    d_germline_start: int = None
    d_germline_end: int = None
    d_alignment_start: int = None
    d_alignment_end: int = None
    d2_sequence_start: int = None
    d2_sequence_end: int = None
    d2_germline_start: int = None
    d2_germline_end: int = None
    d2_alignment_start: int = None
    d2_alignment_end: int = None
    j_sequence_start: int = None
    j_sequence_end: int = None
    j_germline_start: int = None
    j_germline_end: int = None
    j_alignment_start: int = None
    j_alignment_end: int = None
    c_sequence_start: int = None
    c_sequence_end: int = None
    c_germline_start: int = None
    c_germline_end: int = None
    c_alignment_start: int = None
    c_alignment_end: int = None
    cdr1_start: int = None
    cdr1_end: int = None
    cdr2_start: int = None
    cdr2_end: int = None
    cdr3_start: int = None
    cdr3_end: int = None
    fwr1_start: int = None
    fwr1_end: int = None
    fwr2_start: int = None
    fwr2_end: int = None
    fwr3_start: int = None
    fwr3_end: int = None
    fwr4_start: int = None
    fwr4_end: int = None
    v_sequence_alignment: str = None
    v_sequence_alignment_aa: str = None
    d_sequence_alignment: str = None
    d_sequence_alignment_aa: str = None
    d2_sequence_alignment: str = None
    d2_sequence_alignment_aa: str = None
    j_sequence_alignment: str = None
    j_sequence_alignment_aa: str = None
    c_sequence_alignment: str = None
    c_sequence_alignment_aa: str = None
    v_germline_alignment: str = None
    v_germline_alignment_aa: str = None
    d_germline_alignment: str = None
    d_germline_alignment_aa: str = None
    d2_germline_alignment: str = None
    d2_germline_alignment_aa: str = None
    j_germline_alignment: str = None
    j_germline_alignment_aa: str = None
    c_germline_alignment: str = None
    c_germline_alignment_aa: str = None
    junction_length: int = None
    junction_aa_length: int = None
    np1_length: int = None
    np2_length: int = None
    np3_length: int = None
    n1_length: int = None
    n2_length: int = None
    n3_length: int = None
    p3v_length: int = None
    p5d_length: int = None
    p3d_length: int = None
    p5d2_length: int = None
    p3d2_length: int = None
    p5j_length: int = None
    v_frameshift: bool = None
    j_frameshift: bool = None
    d_frame: int = None
    d2_frame: int = None
    consensus_count: int = None
    duplicate_count: int = None
    umi_count: int = None
    cell_id: str = None
    clone_id: str = None
    repertoire_id: str = None
    sample_processing_id: str = None
    data_processing_id: str = None
    rearrangement_id: str = None
    rearrangement_set_id: str = None
    germline_database: str = None

    STR_TO_TYPE = {'str': str, 'int': int, 'float': float, 'bool': bool,
                   'AminoAcidEncoding': bnp.encodings.AminoAcidEncoding,
                   'DNAEncoding': bnp.encodings.DNAEncoding}

    TYPE_TO_STR = {**{val: key for key, val in STR_TO_TYPE.items()},
                   **{AlphabetEncoding('ACDEFGHIKLMNPQRSTVWY*'): 'AminoAcidEncoding',
                      AlphabetEncoding('ACGT'): 'DNAEncoding'}}

    @classmethod
    def get_neutral_value(cls, field_type):
        neutral_values = {str: '', int: -1, DNAEncoding: '', AminoAcidEncoding: '', float: -1.,
                          bool: True}
        return neutral_values[field_type]

