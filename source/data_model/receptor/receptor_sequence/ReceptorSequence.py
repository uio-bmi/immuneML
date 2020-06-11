# quality: gold
from source.data_model.DatasetItem import DatasetItem
from source.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.SequenceType import SequenceType


class ReceptorSequence(DatasetItem):

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

    def get_sequence(self):
        """
        :return: receptor_sequence (nucleotide/amino acid) that corresponds to preset
        receptor_sequence type from EnvironmentSettings class
        """
        if EnvironmentSettings.get_sequence_type() == SequenceType.AMINO_ACID:
            return self.amino_acid_sequence
        else:
            return self.nucleotide_sequence

    def set_sequence(self, sequence: str, sequence_type: SequenceType):
        if sequence_type == SequenceType.AMINO_ACID:
            self.amino_acid_sequence = sequence
        else:
            self.nucleotide_sequence = sequence

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
            raise KeyError("ReceptorSequence does not have attribute {}.".format(name))
