# quality: gold
import os

from source.environment.SequenceType import SequenceType


class EnvironmentSettings:
    """
    Class containing environment variables, like receptor_sequence type,
    root path etc.
    """

    sequence_type = SequenceType.AMINO_ACID
    root_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
    default_params_path = root_path + "config/default_params/"
    tmp_test_path = root_path + "test/tmp/"
    max_sequence_length = 20

    @staticmethod
    def set_sequence_type(sequence_type: SequenceType):
        EnvironmentSettings.sequence_type = sequence_type

    @staticmethod
    def get_sequence_type() -> SequenceType:
        return EnvironmentSettings.sequence_type

    @staticmethod
    def get_sequence_alphabet():
        """
        :return: alphabetically sorted receptor_sequence alphabet
        """
        if EnvironmentSettings.sequence_type == SequenceType.AMINO_ACID:
            alphabet = list("ACDEFGHIKLMNPQRSTVWY")
            alphabet.sort()
        else:
            alphabet = list("ACGT")
            alphabet.sort()
        return alphabet
