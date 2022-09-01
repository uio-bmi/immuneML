import os
import shutil
import numpy as np
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.PositionalMotifEncoder import PositionalMotifEncoder
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.encodings.motif_encoding.WeightHelper import WeightHelper
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestWeightHelper(TestCase):

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

        self.assertDictEqual(WeightHelper.compute_column_contributions(column, pseudocount_value=0), correct_result)

    def test_compute_positional_aa_contributions(self):
        np_sequences = np.asarray(['A' 'A', 'Y' 'A', 'Y' 'Y', 'C' 'Y', 'C' 'Y']).view('U1').reshape(5, -1)

        correct_result = {0: {**{aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}, **{"A": 0.2, "Y": 0.4, "C": 0.4}},
                          1: {**{aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}, **{"A": 0.4, "Y": 0.6}}}

        result = WeightHelper.compute_positional_aa_contributions(np_sequences, pseudocount_value=0)

        self.assertDictEqual(result, correct_result)

    def test_compute_sequence_weight(self):
        positional_weights = {0: {"A": 0.2, "B": 0.4, "C": 0.4}, 1: {"A": 0.4, "B": 0.6}}
        alphabet = "ABCD"

        self.assertEqual(WeightHelper.compute_sequence_weight("AA", positional_weights, alphabet=alphabet), 0.25 / 0.2 * 0.25 / 0.4)
        self.assertEqual(WeightHelper.compute_sequence_weight("AB", positional_weights, alphabet=alphabet), 0.25 / 0.2 * 0.25 / 0.6)
        self.assertEqual(WeightHelper.compute_sequence_weight("BA", positional_weights, alphabet=alphabet), 0.25 / 0.4 * 0.25 / 0.4)
        self.assertEqual(WeightHelper.compute_sequence_weight("BB", positional_weights, alphabet=alphabet), 0.25 / 0.4 * 0.25 / 0.6)
        self.assertEqual(WeightHelper.compute_sequence_weight("CA", positional_weights, alphabet=alphabet), 0.25 / 0.4 * 0.25 / 0.4)
        self.assertEqual(WeightHelper.compute_sequence_weight("CB", positional_weights, alphabet=alphabet), 0.25 / 0.4 * 0.25 / 0.6)


