# quality: gold
import os
from pathlib import Path

from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class EnvironmentSettings:
    """
    Class containing environment variables, like receptor_sequence type,
    root path etc.
    """

    sequence_type = SequenceType.AMINO_ACID
    root_path = Path(os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../../") + "/")
    default_params_path = root_path / "immuneML/config/default_params"
    tmp_test_path = root_path / "test/tmp"
    default_analysis_path = root_path / "analysis_runs"
    cache_path = root_path / "cache"
    tmp_cache_path = tmp_test_path / "cache"
    html_templates_path = root_path / "immuneML/presentation/html/templates"
    specs_docs_path = root_path / "docs/specs"
    source_docs_path = root_path / "docs/source"
    max_sequence_length = 20
    low_memory = True
    max_points_on_plot = 10000
    compairr_paths = [Path("/usr/local/bin/compairr"), Path("./compairr/src/compairr"),
                      root_path / 'compairr/src/compairr', root_path / 'compairr']

    @staticmethod
    def reset_cache_path():
        EnvironmentSettings.cache_path = EnvironmentSettings.root_path / "cache"
        del os.environ[Constants.CACHE_PATH]

    @staticmethod
    def set_cache_path(path: Path):
        EnvironmentSettings.cache_path = Path(path)
        PathBuilder.build(path)
        os.environ[Constants.CACHE_PATH] = str(EnvironmentSettings.cache_path)
        print_log(f"Setting temporary cache path to {path}", include_datetime=True)

    @staticmethod
    def get_cache_type():
        if Constants.CACHE_TYPE not in os.environ:
            os.environ[Constants.CACHE_TYPE] = CacheType.PRODUCTION.name
        return CacheType[os.environ[Constants.CACHE_TYPE].upper()]

    @staticmethod
    def get_cache_path(cache_type: CacheType = None):
        cache_type = EnvironmentSettings.get_cache_type() if cache_type is None else cache_type
        if cache_type == CacheType.PRODUCTION:
            return EnvironmentSettings.cache_path if Constants.CACHE_PATH not in os.environ else Path(os.environ[Constants.CACHE_PATH])
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
    def get_sequence_alphabet(sequence_type: SequenceType = None):
        """
        :return: alphabetically sorted receptor_sequence alphabet
        """
        seq_type = sequence_type if sequence_type is not None else EnvironmentSettings.sequence_type
        if seq_type == SequenceType.AMINO_ACID:
            alphabet = list("ACDEFGHIKLMNPQRSTVWY")
            alphabet.sort()
        elif seq_type == SequenceType.NUCLEOTIDE:
            alphabet = list("ACGT")
            alphabet.sort()
        else:
            raise RuntimeError("EnvironmentSettings: the sequence alphabet cannot be obtained if sequence_type was "
                               "not set properly. Expected AMINO_ACID or NUCLEOTIDE, but got {seq_type} instead.")
        return alphabet
