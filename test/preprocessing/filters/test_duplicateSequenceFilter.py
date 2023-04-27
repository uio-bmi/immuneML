import os
import shutil
from unittest import TestCase

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


class TestDuplicateSequenceFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "duplicatesequencefilter/")

        dataset = RepertoireDataset(repertoires=[Repertoire.build(sequence_aa=["AAA", "AAA", "CCC", "AAA", "CCC", "CCC", "CCC"],
                                                                  sequence=["ntAAA", "ntBBB", "ntCCC", "ntAAA", "ntCCC", "ntCCC", "ntDDD"],
                                                                  v_call=["v1", "v1", "v1", "v1", "v1", "v1", "v1"],
                                                                  j_call=["j1", "j1", "j1", "j1", "j1", "j1", "j1"],
                                                                  chain=[Chain.ALPHA, Chain.ALPHA, Chain.ALPHA, Chain.ALPHA, Chain.ALPHA,
                                                                          Chain.ALPHA, Chain.BETA],
                                                                  duplicate_count=[10, 20, 30, 5, 20, None, 40],
                                                                  region_type=["IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3", "IMGT_CDR3"],
                                                                  custom_lists={"custom1": ["yes", "yes", "yes", "no", "no", "no", "no"],
                                                                                "custom2": ["yes", "yes", "yes", "no", "no", "no", "no"]},
                                                                  sequence_id=[1, 2, 3, 4, 5, 6, 7],
                                                                  path=path)])

        # collapse by amino acids & use sum counts
        dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.AMINO_ACID,
                                            count_agg=CountAggregationFunction.SUM, batch_size=1)

        reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

        attr = reduced_repertoire.get_attributes(["sequence_id", "sequence_aa", "sequence", "duplicate_count", "chain"])

        self.assertEqual(3, len(reduced_repertoire.get_sequence_identifiers()))
        self.assertListEqual(["AAA", "CCC", "CCC"], list(attr["sequence_aa"]))
        self.assertListEqual(["ntAAA", "ntCCC", "ntDDD"], list(attr["sequence"]))
        self.assertListEqual([35, 50, 40], list(attr["duplicate_count"]))
        self.assertListEqual([1, 3, 7], list(attr["sequence_id"]))
        self.assertListEqual(['ALPHA', 'ALPHA', 'BETA'], list(attr["chain"]))

        # collapse by nucleotides & use min counts
        dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.NUCLEOTIDE,
                                            count_agg=CountAggregationFunction.MIN, batch_size=4)

        reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

        attr = reduced_repertoire.get_attributes(["sequence_id", "sequence_aa", "sequence", "duplicate_count"])

        self.assertEqual(4, len(reduced_repertoire.get_sequence_identifiers()))
        self.assertListEqual([1, 2, 3, 7], list(attr["sequence_id"]))
        self.assertListEqual(["AAA", "AAA", "CCC", "CCC"], list(attr["sequence_aa"]))
        self.assertListEqual(["ntAAA", "ntBBB", "ntCCC", "ntDDD"], list(attr["sequence"]))
        self.assertListEqual([5, 20, 20, 40], list(attr["duplicate_count"]))

        shutil.rmtree(path)
