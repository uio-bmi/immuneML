import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.SequenceType import SequenceType
from source.preprocessing.filters.CountAggregationFunction import CountAggregationFunction
from source.preprocessing.filters.DuplicateSequenceFilter import DuplicateSequenceFilter
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestDuplicateSequenceFilter(TestCase):
    def test_process(self):
        path = EnvironmentSettings.root_path + "test/tmp/duplicatesequencefilter/"
        PathBuilder.build(path)

        dataset = RepertoireDataset(repertoires=[SequenceRepertoire.build(sequence_aas=["AAA", "AAA", "CCC", "AAA", "CCC", "CCC", "CCC"],
                                                                          sequences=["ntAAA", "ntBBB", "ntCCC", "ntAAA", "ntCCC", "ntCCC", "ntDDD"],
                                                                          v_genes=["v1", "v1", "v1", "v1", "v1", "v1", "v1"],
                                                                          j_genes=["j1", "j1", "j1", "j1", "j1", "j1", "j1"],
                                                                          chains=[Chain.A, Chain.A, Chain.A, Chain.A, Chain.A, Chain.A, Chain.B],
                                                                          counts=[10, 20, 30, 5, 20, None, 40],
                                                                          region_types=["CDR3", "CDR3", "CDR3", "CDR3", "CDR3", "CDR3", "CDR3"],
                                                                          custom_lists={"custom1": ["yes", "yes", "yes", "no", "no", "no", "no"],
                                                                                        "custom2": ["yes", "yes", "yes", "no", "no", "no", "no"]},
                                                                          sequence_identifiers=[1, 2, 3, 4, 5, 6, 7],
                                                                          path=path)])

        # collapse by amino acids & use sum counts
        dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.AMINO_ACID,
                                            count_agg=CountAggregationFunction.SUM, batch_size=4)

        reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

        attr = reduced_repertoire.get_attributes(["sequence_identifiers", "sequence_aas", "sequences", "counts", "chains"])

        self.assertEqual(3, len(reduced_repertoire.get_sequence_identifiers()))
        self.assertListEqual(["AAA", "CCC", "CCC"], list(attr["sequence_aas"]))
        self.assertListEqual(["ntAAA", "ntCCC", "ntDDD"], list(attr["sequences"]))
        self.assertListEqual([35, 50, 40], list(attr["counts"]))
        self.assertListEqual([1, 3, 7], list(attr["sequence_identifiers"]))
        self.assertListEqual([Chain("A"), Chain("A"), Chain("B")], list(attr["chains"]))


        # collapse by nucleotides & use min counts
        dupfilter = DuplicateSequenceFilter(filter_sequence_type=SequenceType.NUCLEOTIDE,
                                            count_agg=CountAggregationFunction.MIN, batch_size=4)

        reduced_repertoire = dupfilter.process_dataset(dataset=dataset, result_path=path).repertoires[0]

        attr = reduced_repertoire.get_attributes(["sequence_identifiers", "sequence_aas", "sequences", "counts"])

        self.assertEqual(4, len(reduced_repertoire.get_sequence_identifiers()))
        self.assertListEqual([1, 2, 3, 7], list(attr["sequence_identifiers"]))
        self.assertListEqual(["AAA", "AAA", "CCC", "CCC"], list(attr["sequence_aas"]))
        self.assertListEqual(["ntAAA", "ntBBB", "ntCCC", "ntDDD"], list(attr["sequences"]))
        self.assertListEqual([5, 20, 20, 40], list(attr["counts"]))


        shutil.rmtree(path)
