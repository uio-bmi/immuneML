from dataclasses import fields

import bionumpy as bnp
from bionumpy import DNAEncoding, AminoAcidEncoding
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy.encodings import AlphabetEncoding

AminoAcidXEncoding = AlphabetEncoding('ACDEFGHIKLMNPQRSTVWXY*')


@bnpdataclass
class AIRRSequenceSet:
    sequence_id: str = ''
    sequence: DNAEncoding = ''
    quality: str = ''
    sequence_aa: AminoAcidXEncoding = ''
    rev_comp: str = 'False'
    productive: str = 'False'
    vj_in_frame: str = 'False'
    stop_codon: str = 'False'
    complete_vdj: str = 'False'
    locus: str = ''
    locus_species: str = ''
    v_call: str = ''
    d_call: str = ''
    d2_call: str = ''
    j_call: str = ''
    c_call: str = ''
    sequence_alignment: str = ''
    quality_alignment: str = ''
    sequence_alignment_aa: str = ''
    germline_alignment: str = ''
    germline_alignment_aa: str = ''
    junction: DNAEncoding = None
    junction_aa: AminoAcidXEncoding = None
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
    cdr3_aa: AminoAcidXEncoding = None
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
    v_cigar: str = ''
    d_score: float = None
    d_identity: float = None
    d_support: float = None
    d_cigar: str = ''
    d2_score: float = None
    d2_identity: float = None
    d2_support: float = None
    d2_cigar: str = ''
    j_score: float = None
    j_identity: float = None
    j_support: float = None
    j_cigar: str = ''
    c_score: float = None
    c_identity: float = None
    c_support: float = None
    c_cigar: str = ''
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
    v_sequence_alignment: str = ''
    v_sequence_alignment_aa: str = ''
    d_sequence_alignment: str = ''
    d_sequence_alignment_aa: str = ''
    d2_sequence_alignment: str = ''
    d2_sequence_alignment_aa: str = ''
    j_sequence_alignment: str = ''
    j_sequence_alignment_aa: str = ''
    c_sequence_alignment: str = ''
    c_sequence_alignment_aa: str = ''
    v_germline_alignment: str = ''
    v_germline_alignment_aa: str = ''
    d_germline_alignment: str = ''
    d_germline_alignment_aa: str = ''
    d2_germline_alignment: str = ''
    d2_germline_alignment_aa: str = ''
    j_germline_alignment: str = ''
    j_germline_alignment_aa: str = ''
    c_germline_alignment: str = ''
    c_germline_alignment_aa: str = ''
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
    v_frameshift: str = 'False'
    j_frameshift: str = 'False'
    d_frame: int = None
    d2_frame: int = None
    consensus_count: int = None
    duplicate_count: int = None
    umi_count: int = None
    cell_id: str = ''
    clone_id: str = ''
    repertoire_id: str = ''
    sample_processing_id: str = ''
    data_processing_id: str = ''
    rearrangement_id: str = ''
    rearrangement_set_id: str = ''
    germline_database: str = ''

    STR_TO_TYPE = {'str': str, 'int': int, 'float': float,
                   'AminoAcidEncoding': bnp.encodings.AminoAcidEncoding,
                   'AminoAcidXEncoding': AminoAcidXEncoding,
                   'DNAEncoding': bnp.encodings.DNAEncoding}

    TYPE_TO_STR = {**{val: key for key, val in STR_TO_TYPE.items()},
                   **{AlphabetEncoding('ACDEFGHIKLMNPQRSTVWY*'): 'AminoAcidEncoding',
                      AlphabetEncoding('ACGT'): 'DNAEncoding',
                      AlphabetEncoding('ACDEFGHIKLMNPQRSTVWXY*'): 'AminoAcidXEncoding'}}

    @classmethod
    def get_neutral_value(cls, field_type):
        neutral_values = {str: '', int: -1, DNAEncoding: '', AminoAcidEncoding: '', AminoAcidXEncoding: '', float: -1.}
        return neutral_values[field_type]

    @classmethod
    def get_field_type_dict(cls):
        return {f.name: f.type for f in fields(cls)}
