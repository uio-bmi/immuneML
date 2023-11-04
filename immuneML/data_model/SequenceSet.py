from typing import Any, Dict

import bionumpy as bnp
from bionumpy import AminoAcidEncoding, DNAEncoding
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy.encodings import AlphabetEncoding


@bnpdataclass
class SequenceSet:
    sequence_aa: AminoAcidEncoding = None
    sequence: DNAEncoding = None
    v_call: str = None
    j_call: str = None
    region_type: str = None
    frame_type: str = None
    duplicate_count: int = None
    sequence_id: str = None
    chain: str = None

    STR_TO_TYPE = {'str': str, 'int': int, 'float': float, 'bool': bool,
                   'AminoAcidEncoding': bnp.encodings.AminoAcidEncoding,
                   'DNAEncoding': bnp.encodings.DNAEncoding}

    TYPE_TO_STR = {**{val: key for key, val in STR_TO_TYPE.items()},
                   **{AlphabetEncoding('ACDEFGHIKLMNPQRSTVWY*'): 'AminoAcidEncoding',
                      AlphabetEncoding('ACGT'): 'DNAEncoding'}}

    @classmethod
    def additional_fields_with_types(cls) -> Dict[str, Any]:
        return {'cell_id': str,  'vj_in_frame': int, 'stop_codon': int, 'productive': int, 'rev_comp': int,
                'chain': str}

    @classmethod
    def get_neutral_value(cls, field_type):
        neutral_values = {str: '', int: -1, DNAEncoding: '', AminoAcidEncoding: ''}
        return neutral_values.get(field_type, None)
