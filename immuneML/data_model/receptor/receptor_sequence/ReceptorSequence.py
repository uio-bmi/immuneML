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

    @classmethod
    def create_from_record(cls, record: np.void):
        if 'version' in record.dtype.names and record['version'] == cls.version:
            return ReceptorSequence(**{**{key: record[key] for key, val_type in ReceptorSequence.FIELDS.items()
                                          if val_type == str and key != 'version'},
                                       **{'metadata': SequenceMetadata(**json.loads(record['metadata'])),
                                          'annotation': SequenceAnnotation(**json.loads(record['annotation']))}})
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
