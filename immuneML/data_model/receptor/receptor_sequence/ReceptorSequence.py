# quality: gold
import json

import numpy as np

from immuneML.data_model.DatasetItem import DatasetItem
from immuneML.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.NumpyHelper import NumpyHelper


class ReceptorSequence(DatasetItem):
    FIELDS = {'amino_acid_sequence': str, 'nucleotide_sequence': str, 'identifier': str, 'metadata': dict, 'annotation': dict, 'version': str}
    version = "1"

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
    def create_from_record(cls, record: np.void):
        if 'version' in record.dtype.names and record['version'] == cls.version:
            return ReceptorSequence(**{**{key: record[key] for key, val_type in ReceptorSequence.FIELDS.items()
                                          if val_type == str and key != 'version'},
                                       **{'metadata': SequenceMetadata(**json.loads(record['metadata'])) if record['metadata'] != '' else None,
                                          'annotation': SequenceAnnotation(**json.loads(record['annotation'])) if record['annotation'] != ''
                                          else None}})
        else:
            raise NotImplementedError

    @classmethod
    def get_record_names(cls):
        return [key for key in cls.FIELDS]

    def __init__(self,
                 amino_acid_sequence: str = None,
                 nucleotide_sequence: str = None,
                 identifier: str = None,
                 annotation: SequenceAnnotation = None,
                 metadata: SequenceMetadata = SequenceMetadata()):
        self.identifier = identifier
        self.amino_acid_sequence = amino_acid_sequence
        self.nucleotide_sequence = nucleotide_sequence
        self.annotation = annotation
        self.metadata = metadata

    def set_metadata(self, metadata: SequenceMetadata):
        self.metadata = metadata

    def set_annotation(self, annotation: SequenceAnnotation):
        self.annotation = annotation

    def get_sequence(self, sequence_type: SequenceType = None):
        """Returns receptor_sequence (nucleotide/amino acid) that corresponds to provided sequence type or preset receptor_sequence type from
        EnvironmentSettings class if no type is provided"""

        sequence_type_ = EnvironmentSettings.get_sequence_type() if sequence_type is None else sequence_type
        if sequence_type_ == SequenceType.AMINO_ACID:
            return self.amino_acid_sequence
        else:
            return self.nucleotide_sequence

    def set_sequence(self, sequence: str, sequence_type: SequenceType):
        if sequence_type == SequenceType.AMINO_ACID:
            self.amino_acid_sequence = sequence
        else:
            self.nucleotide_sequence = sequence
            self.amino_acid_sequence = self._convert_to_aa(sequence)

    def _convert_to_aa(self, nt_sequence: str) -> str:
        kmer_length = 3
        kmers = [nt_sequence[i:i + kmer_length] for i in range(0, len(nt_sequence), kmer_length)]
        return "".join([ReceptorSequence.nt_to_aa_map[kmer] for kmer in kmers])

    def get_record(self):
        """exports the sequence object as a numpy record"""
        return [NumpyHelper.get_numpy_representation(getattr(self, name)) if hasattr(self, name) else getattr(ReceptorSequence, name)
                for name in ReceptorSequence.FIELDS.keys()]

    def get_attribute(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.metadata, name):
            return getattr(self.metadata, name)
        elif name in self.metadata.custom_params:
            return self.metadata.custom_params[name]
        elif hasattr(self.annotation, name):
            return getattr(self.annotation, name)
        else:
            raise KeyError(f"ReceptorSequence {self.identifier} does not have attribute {name}.")
