import os
import shutil

import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.repertoire.Repertoire import Repertoire
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
        repertoires=[Repertoire.build(sequence_aa=["AAA", "AAA", "CCC", "AAA", "CCC", "CCC", "CCC"],
                                      sequence=["AAAAA", "CCAAA", "AACCC", "AAAAA", "AACCC", "AACCC", "AATTT"],
                                      v_call=["v1", "v1", "v1", "v1", "v1", "v1", "v1"],
                                      j_call=["j1", "j1", "j1", "j1", "j1", "j1", "j1"],
                                      chain=[Chain.ALPHA, Chain.ALPHA, Chain.ALPHA, Chain.ALPHA, Chain.ALPHA,
                                             Chain.ALPHA, Chain.BETA],
                                      duplicate_count=[10, 20, 30, 5, 20, None, 40],
                                      region_type=["IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3",
                                                   "IMGT_CDR3", "IMGT_CDR3"],
                                      custom1=["yes", "yes", "yes", "no", "no", "no", "no"],
                                      custom2=["yes", "yes", "yes", "no", "no", "no", "no"],
                                      sequence_id=[1, 2, 3, 4, 5, 6, 7],
                                      path=path)])

    # collapse by amino acids & use sum counts
    dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.AMINO_ACID,
                                        count_agg=CountAggregationFunction.SUM, batch_size=1)

    reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

    attr = reduced_repertoire.get_attributes(["sequence_id", "sequence_aa", "sequence", "duplicate_count", "chain"],
                                             as_list=True)

    assert 3 == len(reduced_repertoire.get_sequence_identifiers())
    assert ["AAA", "CCC", "CCC"] == attr["sequence_aa"]
    assert ["AAAAA", "AACCC", "AATTT"] == attr["sequence"]
    assert [35, 50, 40] == attr["duplicate_count"]
    assert ['1', '3', '7'] == attr["sequence_id"]
    assert ['ALPHA', 'ALPHA', 'BETA'] == attr["chain"]

    # collapse by nucleotides & use min counts
    dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.NUCLEOTIDE,
                                        count_agg=CountAggregationFunction.MIN, batch_size=4)

    reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

    attr = reduced_repertoire.get_attributes(["sequence_id", "sequence_aa", "sequence", "duplicate_count"], as_list=True)

    assert 4 == len(reduced_repertoire.get_sequence_identifiers())
    assert ['1', '2', '3', '7'] == attr["sequence_id"]
    assert ["AAA", "AAA", "CCC", "CCC"] == attr["sequence_aa"]
    assert ["AAAAA", "CCAAA", "AACCC", "AATTT"] == attr["sequence"]
    assert [5, 20, 20, 40] == attr["duplicate_count"]

    shutil.rmtree(path)
