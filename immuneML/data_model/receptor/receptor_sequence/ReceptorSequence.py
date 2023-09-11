# quality: gold
from typing import List
from uuid import uuid4

from immuneML.data_model.DatasetItem import DatasetItem
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType


class ReceptorSequence(DatasetItem):
    FIELDS = {'sequence_aa': str, 'sequence': str, 'sequence_id': str, 'metadata': dict, 'version': str}
    version = "2"

    nt_to_aa_map = {
        "AAA": "K", "AAC": "N", "AAG": "K", "AAT": "N", "ACA": "T", "ACC": "T", "ACG": "T", 'ACT': 'T',
        'AGA': 'R', "AGC": "S", "AGG": "R", "AGT": "S", "ATA": "I", "ATC": "I", 'ATG': "M", "ATT": "I",
        "CAA": "Q", "CAC": "H", "CAG": "Q", 'CAT': "H", "CCA": "P", "CCC": "P", "CCG": "P", 'CCT': "P",
        "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R", "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
        "GAA": "E", "GAC": "D", "GAG": "E", 'GAT': "D", "GCA": "A", "GCC": "A", 'GCG': "A", "GCT": "A",
        "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G", "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
        "TAA": None, "TAC": "Y", "TAG": None, "TAT": "Y", "TCA": "S", "TCC": 'S', "TCG": "S", "TCT": "S",
        "TGA": None, "TGC": "C", "TGG": "W", "TGT": "C", "TTA": "L", "TTC": "F", "TTG": "L", "TTT": "F"
    }

    @classmethod
    def create_from_record(cls, **kwargs):
        metadata_keys = vars(SequenceMetadata())
        metadata_obj = SequenceMetadata(**{**{key: val for key, val in kwargs.items() if key in metadata_keys},
                                           **{'custom_params': {key: value for key, value in kwargs.items()
                                                                if key not in metadata_keys and key not in ReceptorSequence.FIELDS}}})
        return ReceptorSequence(**{**{key: kwargs[key] for key, val_type in ReceptorSequence.FIELDS.items()
                                      if val_type == str and key != 'version'},
                                   **{'metadata': metadata_obj}})

    @classmethod
    def get_record_names(cls):
        return [key for key in cls.FIELDS]

    def __init__(self,
                 sequence_aa: str = None,
                 sequence: str = None,
                 sequence_id: str = None,
                 metadata: SequenceMetadata = None):
        self.sequence_id = sequence_id if sequence_id is not None and sequence_id != "" else uuid4().hex
        self.sequence_aa = sequence_aa
        self.sequence = sequence
        self.metadata = metadata if metadata is not None else SequenceMetadata()

    @property
    def identifier(self):
        return self.sequence_id

    def __repr__(self):
        return f"ReceptorSequence(sequence_aa={self.sequence_aa}, sequence={self.sequence}, " \
               f"sequence_id={self.sequence_id}, " \
               f"metadata={vars(self.metadata) if self.metadata is not None else '{}'})"

    def set_metadata(self, metadata: SequenceMetadata):
        self.metadata = metadata

    def get_sequence(self, sequence_type: SequenceType = None):
        """Returns receptor_sequence (nucleotide/amino acid) that corresponds to provided sequence type or preset receptor_sequence type from
        EnvironmentSettings class if no type is provided"""

        sequence_type_ = EnvironmentSettings.get_sequence_type() if sequence_type is None else sequence_type
        if sequence_type_ == SequenceType.AMINO_ACID:
            return self.sequence_aa
        else:
            return self.sequence

    def set_sequence(self, sequence: str, sequence_type: SequenceType):
        if sequence_type == SequenceType.AMINO_ACID:
            self.sequence_aa = sequence
        else:
            self.sequence = sequence
            self.sequence_aa = self._convert_to_aa(sequence)

    def get_id(self):
        return self.sequence_id

    def _convert_to_aa(self, nt_sequence: str) -> str:
        return ReceptorSequence.nt_to_aa(nt_sequence)

    @classmethod
    def nt_to_aa(cls, nt_sequence: str):
        kmer_length = 3
        kmers = [nt_sequence[i:i + kmer_length] for i in range(0, len(nt_sequence), kmer_length)]
        return "".join([ReceptorSequence.nt_to_aa_map[kmer] for kmer in kmers])

    def get_all_attribute_names(self) -> List[str]:
        return [el for el in vars(self) if el not in ['metadata']] + self.metadata.get_all_attribute_names() \
            if self.metadata is not None else []

    def get_attribute(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.metadata, name):
            return self.metadata.get_attribute(name)
        elif name in self.metadata.custom_params:
            return self.metadata.custom_params[name]
        else:
            return ''
