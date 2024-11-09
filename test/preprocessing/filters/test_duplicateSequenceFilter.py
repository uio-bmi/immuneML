import os
import shutil

import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from immuneML.preprocessing.filters.DuplicateSequenceFilter import DuplicateSequenceFilter
from immuneML.util.PathBuilder import PathBuilder


def test_duplicate_seq_filter():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "duplicate_sequence_filter/")

    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    dataset = RepertoireDataset(
        repertoires=[Repertoire.build(cdr3_aa=["AAA", "AAA", "CCC", "AAA", "CCC", "CCC", "CCC"],
                                      cdr3=["AAAAA", "CCAAA", "AACCC", "AAAAA", "AACCC", "AACCC", "AATTT"],
                                      v_call=["v1", "v1", "v1", "v1", "v1", "v1", "v1"],
                                      j_call=["j1", "j1", "j1", "j1", "j1", "j1", "j1"],
                                      locus=[Chain.ALPHA.value, Chain.ALPHA.value, Chain.ALPHA.value, Chain.ALPHA.value, Chain.ALPHA.value,
                                             Chain.ALPHA.value, Chain.BETA.value],
                                      duplicate_count=[10, 20, 30, 5, 20, 10, 40],
                                      custom1=["yes", "yes", "yes", "no", "no", "no", "no"],
                                      custom2=["yes", "yes", "yes", "no", "no", "no", "no"],
                                      sequence_id=['1', '2', '3', '4', '5', '6', '7'],
                                      path=path,
                                      metadata={})])

    # collapse by amino acids & use sum counts

    params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "preprocessing", "duplicate_sequence_filter")
    dupfilter = DuplicateSequenceFilter.build_object(**params)

    reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

    attr = reduced_repertoire.data.topandas()[["sequence_id", "cdr3_aa", "cdr3", "duplicate_count", "locus"]]

    assert 3 == attr.shape[0]
    assert all(["AAA", "CCC", "CCC"] == attr["cdr3_aa"])
    assert all(["AAAAA", "AACCC", "AATTT"] == attr["cdr3"])
    assert all([35, 60, 40] == attr["duplicate_count"])
    assert all(['1', '3', '7'] == attr["sequence_id"])
    assert all(['TRA', 'TRA', 'TRB'] == attr["locus"])

    # collapse by nucleotides & use min counts
    dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.NUCLEOTIDE,
                                        count_agg=CountAggregationFunction.MIN)

    reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

    attr = reduced_repertoire.data.topandas()[["sequence_id", "cdr3_aa", "cdr3", "duplicate_count"]]

    assert 4 == attr.shape[0]
    assert all(['1', '2', '3', '7'] == attr["sequence_id"])
    assert all(["AAA", "AAA", "CCC", "CCC"] == attr["cdr3_aa"])
    assert all(["AAAAA", "CCAAA", "AACCC", "AATTT"] == attr["cdr3"])
    assert all([5, 20, 10, 40] == attr["duplicate_count"])

    shutil.rmtree(path)
