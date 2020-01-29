# quality: gold
import os

from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.SequenceType import SequenceType
from source.logging.LogLevel import LogLevel


class EnvironmentSettings:
    """
    Class containing environment variables, like receptor_sequence type,
    root path etc.
    """

    sequence_type = SequenceType.AMINO_ACID
    root_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../../") + "/"
    default_params_path = root_path + "config/default_params/"
    tmp_test_path = root_path + "test/tmp/"
    default_analysis_path = root_path + "analysis_runs/"
    cache_path = root_path + "cache/"
    visualization_path = root_path + "source/visualization/"
    tmp_cache_path = tmp_test_path + "cache/"
    max_sequence_length = 20
    log_level = LogLevel.DEBUG
    low_memory = True

    @staticmethod
    def get_cache_path(cache_type: CacheType = None):
        cache_type = CacheType[os.environ[Constants.CACHE_TYPE].upper()] if cache_type is None else cache_type
        if cache_type == CacheType.PRODUCTION:
            return EnvironmentSettings.cache_path
        elif cache_type == CacheType.TEST:
            return EnvironmentSettings.tmp_cache_path
        else:
            raise RuntimeError("Cache is not set up.")

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
