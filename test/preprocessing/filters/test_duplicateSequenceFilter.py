import os
import shutil
from unittest import TestCase

import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.SequenceSet import Repertoire, ReceptorSequence
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from immuneML.preprocessing.filters.DuplicateSequenceFilter import DuplicateSequenceFilter
from immuneML.util.PathBuilder import PathBuilder


class TestDuplicateSequenceFilter(TestCase):

    def test_duplicate_seq_filter_rep_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "duplicate_sequence_filter_rep_dataset/")

        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

        dataset = RepertoireDataset(
            repertoires=[Repertoire.build(cdr3_aa=["AAA", "AAA", "CCC", "AAA", "CCC", "CCC", "CCC"],
                                          cdr3=["AAAAA", "CCAAA", "AACCC", "AAAAA", "AACCC", "AACCC", "AATTT"],
                                          v_call=["v1", "v1", "v1", "v1", "v1", "v1", "v1"],
                                          j_call=["j1", "j1", "j1", "j1", "j1", "j1", "j1"],
                                          locus=[Chain.ALPHA.value, Chain.ALPHA.value, Chain.ALPHA.value, Chain.ALPHA.value, Chain.ALPHA.value,
                                                 Chain.ALPHA.value, Chain.BETA.value],
                                          duplicate_count=[10, 20, 30, 5, 20, -1, 40],
                                          custom1=["yes", "yes", "yes", "no", "no", "no", "no"],
                                          custom2=["yes", "yes", "yes", "no", "no", "no", "no"],
                                          sequence_id=['1', '2', '3', '4', '5', '6', '7'],
                                          path=path,
                                          metadata={})])

        # collapse by amino acids & use sum counts
        dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.AMINO_ACID,
                                            count_agg=CountAggregationFunction.SUM, batch_size=1)

        reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

        attr = reduced_repertoire.data.topandas()[["sequence_id", "cdr3_aa", "cdr3", "duplicate_count", "locus"]]

        assert 3 == attr.shape[0]
        assert all(["AAA", "CCC", "CCC"] == attr["cdr3_aa"])
        assert all(["AAAAA", "AACCC", "AATTT"] == attr["cdr3"])
        assert all([35, 50, 40] == attr["duplicate_count"])
        assert all(['1', '3', '7'] == attr["sequence_id"])
        assert all(['TRA', 'TRA', 'TRB'] == attr["locus"])

        # collapse by nucleotides & use min counts
        dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.NUCLEOTIDE,
                                            count_agg=CountAggregationFunction.MIN, batch_size=4)

        reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

        attr = reduced_repertoire.data.topandas()[["sequence_id", "cdr3_aa", "cdr3", "duplicate_count"]]

        assert 4 == attr.shape[0]
        assert all(['1', '2', '3', '7'] == attr["sequence_id"])
        assert all(["AAA", "AAA", "CCC", "CCC"] == attr["cdr3_aa"])
        assert all(["AAAAA", "CCAAA", "AACCC", "AATTT"] == attr["cdr3"])
        assert all([5, 20, 20, 40] == attr["duplicate_count"])

        shutil.rmtree(path)

    def test_duplicate_seq_filter_seq_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "duplicate_sequence_filter_seq_dataset/")

        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

        sequences = [ReceptorSequence(sequence_aa="AAACCC", sequence="AAACCC", sequence_id="1",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="ACACAC", sequence="ACACAC", sequence_id="2",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="CCCAAA", sequence="CCCAAA", sequence_id="3",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="AAACCC", sequence="AAACCC", sequence_id="4",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="ACACAC", sequence="ACACAC", sequence_id="2",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="CCCAAA", sequence="CCCAAA", sequence_id="6",
                                      metadata={"l1": 2})]

        dataset = SequenceDataset.build_from_objects(sequences, PathBuilder.build(path / 'data'), 'dup_seq_dataset')

        dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.AMINO_ACID,
                                            count_agg=CountAggregationFunction.SUM, batch_size=1)

        reduced_sequence_dataset = dupfilter.process_dataset(dataset=dataset, result_path=path)

        attr = reduced_sequence_dataset.data.topandas()[["sequence_id", "cdr3_aa", "cdr3", "duplicate_count"]]

        assert attr.shape[0] == 3
        assert all(["AAACCC", "ACACAC", "CCCAAA"] == attr["cdr3_aa"])
        assert all(['1', '2', '3'] == attr["sequence_id"])

        shutil.rmtree(path)

