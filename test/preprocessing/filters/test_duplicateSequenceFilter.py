import os
import shutil

import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceParams import Chain, ChainPair
from immuneML.data_model.SequenceSet import Repertoire, ReceptorSequence, Receptor
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from immuneML.preprocessing.filters.DuplicateSequenceFilter import DuplicateSequenceFilter
from immuneML.util.PathBuilder import PathBuilder


def test_duplicate_seq_filter_repertoire():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "duplicate_sequence_filter_rep/")

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
    dupfilter = DuplicateSequenceFilter(groupby_fields=["cdr3"],
                                        count_agg=CountAggregationFunction.MIN)

    reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

    attr = reduced_repertoire.data.topandas()[["sequence_id", "cdr3_aa", "cdr3", "duplicate_count"]]

    assert 4 == attr.shape[0]
    assert all(['1', '2', '3', '7'] == attr["sequence_id"])
    assert all(["AAA", "AAA", "CCC", "CCC"] == attr["cdr3_aa"])
    assert all(["AAAAA", "CCAAA", "AACCC", "AATTT"] == attr["cdr3"])
    assert all([5, 20, 10, 40] == attr["duplicate_count"])

    shutil.rmtree(path)

def test_duplicate_seq_filter_sequences():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "duplicate_sequence_filter_seqs/")

    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    # dataset =
    sequences = [
        ReceptorSequence(sequence_aa="AAAA", sequence_id="1", metadata={"l1": 1, "l2": 1}),
        ReceptorSequence(sequence_aa="AAAA", sequence_id="2", metadata={"l1": 2, "l2": 1}),
        ReceptorSequence(sequence_aa="AAAA", sequence_id="3", metadata={"l1": 1, "l2": 2})]

    dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)

    params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "preprocessing",
                                      "duplicate_sequence_filter")
    params["groupby_fields"] += ["l1"]
    dupfilter = DuplicateSequenceFilter.build_object(**params)

    reduced_sequences = dupfilter.process_dataset(dataset=dataset, result_path=path)

    attr = reduced_sequences.data.topandas()[["sequence_id", "cdr3_aa", "cdr3", "duplicate_count"]]

    assert 2 == attr.shape[0]
    assert all(['1', '2'] == attr["sequence_id"])
    assert all(["AAAA", "AAAA"] == attr["cdr3_aa"])
    assert all([2, 1] == attr["duplicate_count"])

    shutil.rmtree(path)

def test_duplicate_seq_filter_receptors():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "duplicate_sequence_filter_receptor/")

    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    # dataset =
    receptors = [Receptor(chain_1=ReceptorSequence(sequence_aa="AAACCC", locus='alpha', cell_id='1'),
                          chain_2=ReceptorSequence(sequence_aa="AAACCC", locus='beta', cell_id="1"),
                          receptor_id="1", cell_id="1", chain_pair=ChainPair.TRA_TRB,
                          metadata={"l1": 1, "l2": 1}),
                 Receptor(chain_1=ReceptorSequence(sequence_aa="AAA", locus='alpha', cell_id="2"),
                          chain_2=ReceptorSequence(sequence_aa="CCC", locus='beta', cell_id="2"),
                          receptor_id="2", cell_id="2", chain_pair=ChainPair.TRA_TRB,
                          metadata={"l1": 1, "l2": 2}),
                 Receptor(chain_1=ReceptorSequence(sequence_aa="AAACCC", locus='alpha', cell_id="3"),
                          chain_2=ReceptorSequence(sequence_aa="AAACCC", locus='beta', cell_id="3"),
                          receptor_id="3", cell_id="3", chain_pair=ChainPair.TRA_TRB,
                          metadata={"l1": 1, "l2": 2}),
                 Receptor(chain_1=ReceptorSequence(sequence_aa="AAA", locus='alpha', cell_id="4"),
                          chain_2=ReceptorSequence(sequence_aa="CCC", locus='beta', cell_id="4"),
                          receptor_id="4", cell_id="4", chain_pair=ChainPair.TRA_TRB,
                          metadata={"l1": 1, "l2": 2})]

    dataset = ReceptorDataset.build_from_objects(receptors=receptors, path=path)

    params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "preprocessing",
                                      "duplicate_sequence_filter")
    params["groupby_fields"] += ["l2"]
    dupfilter = DuplicateSequenceFilter.build_object(**params)

    reduced_sequences = dupfilter.process_dataset(dataset=dataset, result_path=path)

    attr = reduced_sequences.data.topandas()[["receptor_id", "cdr3_aa", "cdr3", "duplicate_count", "l2"]]

    assert 6 == attr.shape[0]
    assert all(['1', '1', '2', '2', '3', '3'] == attr["receptor_id"])
    assert all(["AAACCC", "AAACCC", "AAA", "CCC", "AAACCC", "AAACCC"] == attr["cdr3_aa"])
    assert all([1, 1, 2, 2, 2, 2] == attr["l2"])
    assert all([1, 1, 2, 2, 1, 1] == attr["duplicate_count"])

    shutil.rmtree(path)

