import os
import numpy as np
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.example_weighting.importance_weighting.ImportanceWeightHelper import ImportanceWeightHelper
from immuneML.environment.Constants import Constants
from immuneML.util.PathBuilder import PathBuilder


class TestImportanceWeightHelper(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_dataset(self, path):
        sequences = [ReceptorSequence(amino_acid_sequence="AA", identifier="1",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="CC", identifier="2",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="AC", identifier="3",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="CA", identifier="4",
                                      metadata=SequenceMetadata(custom_params={"l1": 1}))]

        PathBuilder.build(path)
        return SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

    def test_compute_column_contributions(self):
        column = ["A", "Y", "Y", "C", "C"]

        correct_result = {**{aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}, **{"A": 0.2, "Y": 0.4, "C": 0.4}}

        self.assertDictEqual(ImportanceWeightHelper._compute_column_contributions(column, pseudocount_value=0), correct_result)

    def test_compute_positional_aa_contributions(self):
        np_sequences = np.asarray(['A' 'A', 'Y' 'A', 'Y' 'Y', 'C' 'Y', 'C' 'Y']).view('U1').reshape(5, -1)

        correct_result = {0: {**{aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}, **{"A": 0.2, "Y": 0.4, "C": 0.4}},
                          1: {**{aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}, **{"A": 0.4, "Y": 0.6}}}

        result = ImportanceWeightHelper._compute_positional_aa_frequences_np_sequences(np_sequences, pseudocount_value=0)

        self.assertDictEqual(result, correct_result)

    def test_compute_mutagenesis_probability(self):
        positional_weights = {0: {"A": 0.2, "B": 0.4, "C": 0.4}, 1: {"A": 0.4, "B": 0.6}}

        self.assertEqual(ImportanceWeightHelper.compute_mutagenesis_probability("AA", positional_weights), 0.2 * 0.4)
        self.assertEqual(ImportanceWeightHelper.compute_mutagenesis_probability("AB", positional_weights), 0.2 * 0.6)
        self.assertEqual(ImportanceWeightHelper.compute_mutagenesis_probability("BA", positional_weights), 0.4 * 0.4)
        self.assertEqual(ImportanceWeightHelper.compute_mutagenesis_probability("BB", positional_weights), 0.4 * 0.6)
        self.assertEqual(ImportanceWeightHelper.compute_mutagenesis_probability("CA", positional_weights), 0.4 * 0.4)
        self.assertEqual(ImportanceWeightHelper.compute_mutagenesis_probability("CB", positional_weights), 0.4 * 0.6)

    def test_compute_uniform_probability(self):
        self.assertEqual(ImportanceWeightHelper.compute_uniform_probability("ABCD", 5), (1/5)**4)
        self.assertEqual(ImportanceWeightHelper.compute_uniform_probability("ABCDEFG", 8), (1/8)**7)
        self.assertEqual(ImportanceWeightHelper.compute_uniform_probability("AB", 4), (1/4)**2)


