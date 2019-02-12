# quality: gold
from source.environment.SequenceType import SequenceType


class EnvironmentSettings:
    """
    Class containing environment variables, like sequence type,
    root path etc.
    """

    sequence_type = SequenceType.AMINO_ACID
    root_path = ""
    max_sequence_length = 20

    @staticmethod
    def get_max_sequence_length():
        return EnvironmentSettings.max_sequence_length

    @staticmethod
    def set_max_sequence_length(length: int):
        EnvironmentSettings.max_sequence_length = length

    @staticmethod
    def set_sequence_type(sequence_type: SequenceType):
        EnvironmentSettings.sequence_type = sequence_type

    @staticmethod
    def get_sequence_type() -> SequenceType:
        return EnvironmentSettings.sequence_type

    @staticmethod
    def set_root_path(path):
        EnvironmentSettings.root_path = path

    @staticmethod
    def get_root_path():
        return EnvironmentSettings.root_path

    @staticmethod
    def get_sequence_alphabet():
        """
        :return: alphabetically sorted sequence alphabet
        """
        if EnvironmentSettings.sequence_type == SequenceType.AMINO_ACID:
            alphabet = list("ACDEFGHIKLMNPQRSTVWY")
            alphabet.sort()
        else:
            alphabet = list("ACGT")
            alphabet.sort()
        return alphabet
