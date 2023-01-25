import os
import shutil
import numpy as np
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.encodings.motif_encoding.PositionalMotifParams import PositionalMotifParams
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestPositionalMotifHelper(TestCase):

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

    def test_get_numpy_sequence_representation(self):
        path = EnvironmentSettings.tmp_test_path / "positional_motif_sequence_encoder/test_np/"
        dataset = self._prepare_dataset(path = path)
        output = PositionalMotifHelper.get_numpy_sequence_representation(dataset)

        expected = np.asarray(['A' 'A', 'C' 'C', 'A' 'C', 'C' 'A']).view('U1').reshape(4, -1)

        self.assertEqual(output.shape, expected.shape)

        for i in range(len(output)):
            self.assertListEqual(list(output[i]), list(expected[i]))

            for j in range(len(output[i])):
                self.assertEqual(type(output[i][j]), type(expected[i][j]))

        shutil.rmtree(path)

    def test_test_aa(self):
        sequence_array = np.asarray(['A' 'A', 'B' 'B', 'A' 'B', 'B' 'A']).view('U1').reshape(4, -1)

        self.assertListEqual(list(PositionalMotifHelper.test_aa(sequence_array, 0, "A")), [True, False, True, False])
        self.assertListEqual(list(PositionalMotifHelper.test_aa(sequence_array, 1, "A")), [True, False, False, True])
        self.assertListEqual(list(PositionalMotifHelper.test_aa(sequence_array, 0, "B")), [False, True, False, True])
        self.assertListEqual(list(PositionalMotifHelper.test_aa(sequence_array, 1, "B")), [False, True, True, False])

    def test_test_position(self):
        sequence_array = np.asarray(['A' 'A', 'B' 'B', 'A' 'B', 'C' 'A']).view('U1').reshape(4, -1)

        self.assertListEqual(list(PositionalMotifHelper.test_position(sequence_array, 0, "A")), [True, False, True, False])
        self.assertListEqual(list(PositionalMotifHelper.test_position(sequence_array, 0, "AB")), [True, True, True, False])
        self.assertListEqual(list(PositionalMotifHelper.test_position(sequence_array, 0, "BC")), [False, True, False, True])
        self.assertListEqual(list(PositionalMotifHelper.test_position(sequence_array, 0, "ABC")), [True, True, True, True])

    def test_test_motif(self):
        sequence_array = np.asarray(['A' 'A', 'B' 'B', 'A' 'B', 'B' 'A']).view('U1').reshape(4, -1)

        self.assertListEqual(list(PositionalMotifHelper.test_motif(sequence_array, (0, 1), ("A", "B"))), [False, False, True, False])
        self.assertListEqual(list(PositionalMotifHelper.test_motif(sequence_array, (0, 1), ("E", "E"))), [False, False, False, False])
        self.assertListEqual(list(PositionalMotifHelper.test_motif(sequence_array, (0, 1), ("DE", "DE"))), [False, False, False, False])
        self.assertListEqual(list(PositionalMotifHelper.test_motif(sequence_array, (0, 1), ("A", "BA"))), [True, False, True, False])
        self.assertListEqual(list(PositionalMotifHelper.test_motif(sequence_array, (0, 1), ("AB", "AB"))), [True, True, True, True])
        self.assertListEqual(list(PositionalMotifHelper.test_motif(sequence_array, (0, 1), ("C", "AB"))), [False, False, False, False])
        self.assertListEqual(list(PositionalMotifHelper.test_motif(sequence_array, (0, 1), ("AB", "C"))), [False, False, False, False])

    def test_extend_motif(self):
        np_sequences = np.asarray(['A' 'A', 'C' 'C', 'A' 'C', 'C' 'A']).view('U1').reshape(4, -1)

        outcome = PositionalMotifHelper.extend_motif([[0], ["A"]], np_sequences, {0: ["A", "C"], 1: ["C"]}, count_threshold=1)
        self.assertListEqual(outcome, [[[0, 1], ['A', 'C']]])

        outcome = PositionalMotifHelper.extend_motif([[0], ["A"]], np_sequences, {0: ["A", "C"], 1: ["A", "C", "D"]}, count_threshold=1)
        self.assertListEqual(outcome, [[[0, 1], ['A', 'A']], [[0, 1], ['A', 'C']]])

        outcome = PositionalMotifHelper.extend_motif([[0], ["A"]], np_sequences, {0: ["A", "C"], 1: ["A", "C", "D"]}, count_threshold=0)
        self.assertListEqual(outcome, [[[0, 1], ['A', 'A']], [[0, 1], ['A', 'C']], [[0, 1], ['A', 'D']]])

    def test_identify_legal_positional_aas(self):
        np_sequences = np.asarray(['A' 'A', 'C' 'C', 'A' 'C', 'C' 'D']).view('U1').reshape(4, -1)

        outcome = PositionalMotifHelper.identify_legal_positional_aas(np_sequences, count_threshold=1)
        expected = {0: ["A", "C"], 1: ["A", "C", "D"]}
        self.assertDictEqual(expected, outcome)

        outcome = PositionalMotifHelper.identify_legal_positional_aas(np_sequences, count_threshold=2)
        expected = {0: ["A", "C"], 1: ["C"]}
        self.assertDictEqual(expected, outcome)

    def test_compute_all_candidate_motifs(self):
        np_sequences = np.asarray(['A' 'A', 'A' 'A', 'C' 'C']).view('U1').reshape(3, -1)

        outcome = PositionalMotifHelper.compute_all_candidate_motifs(np_sequences, params=PositionalMotifParams(max_positions=1, count_threshold=2))
        expected = [[[0], ["A"]], [[1], ["A"]]]
        self.assertListEqual(outcome, expected)

        outcome = PositionalMotifHelper.compute_all_candidate_motifs(np_sequences, params=PositionalMotifParams(max_positions=1, count_threshold=1))
        expected = [[[0], ["A"]], [[0], ["C"]], [[1], ["A"]], [[1], ["C"]]]
        self.assertListEqual(outcome, expected)

        outcome = PositionalMotifHelper.compute_all_candidate_motifs(np_sequences, params=PositionalMotifParams(max_positions=2, count_threshold=2))
        expected = [[[0], ["A"]], [[1], ["A"]], [[0, 1], ["A", "A"]]]
        self.assertListEqual(outcome, expected)

    def test_readwrite(self):
        path = EnvironmentSettings.tmp_test_path / "positional_motif_sequence_encoder/test_readwrite/"

        original_motifs = [([0], ["A"]), ([1], ["A"]), ([0, 1], ["A", "A"])]
        PositionalMotifHelper.write_motifs_to_file(original_motifs, filepath=path / "motifs.tsv")
        motifs = PositionalMotifHelper.read_motifs_from_file(filepath=path / "motifs.tsv")

        self.assertListEqual(original_motifs, motifs)

        shutil.rmtree(path)

    def test_get_generalized_motifs(self):
        motifs = [[[2, 3, 5], ["A", "A", "A"]], [[2, 3, 5], ["A", "A", "D"]], [[2, 3, 6], ["A", "A", "C"]]]

        result = PositionalMotifHelper.get_generalized_motifs(motifs)
        expected = [[[2, 3, 5], ["A", "A", "AD"]]]

        self.assertListEqual(result, expected)

        motifs = [[[2, 3, 5], ["A", "A", "A"]], [[2, 3, 7], ["A", "A", "D"]], [[2, 3, 6], ["A", "A", "C"]]]

        result = PositionalMotifHelper.get_generalized_motifs(motifs)
        expected = []

        self.assertListEqual(result, expected)

    def test__sort_motifs_by_index(self):
        motifs = [[[1,2], ["A", "A"]], [[1, 2], ["A", "F"]], [[1, 2], ["G", "D"]], [[5, 6], ["A", "A"]], [[6, 7], ["A", "A"]]]
        result = PositionalMotifHelper.sort_motifs_by_index(motifs)
        expected = {(1,2): [["A", "A"], ["A", "F"], ["G", "D"]],
                    (5, 6): [["A", "A"]],
                    (6, 7): [["A", "A"]]}

        self.assertDictEqual(result, expected)

    def test_get_generalized_motifs_for_index(self):
        indices = [2, 3, 5]
        all_motif_amino_acids = [["A", "A", "A"], ["A", "A", "C"], ["A", "A", "D"], ["D", "A", "D"]]

        result = list(PositionalMotifHelper.get_generalized_motifs_for_index(indices, all_motif_amino_acids))
        expected = [[[2, 3, 5], ["AD", "A", "D"]], [[2, 3, 5], ["A", "A", "AC"]], [[2, 3, 5], ["A", "A", "AD"]],
                    [[2, 3, 5], ["A", "A", "CD"]], [[2, 3, 5], ["A", "A", "ACD"]]]

        self.assertListEqual(result, expected)

    def test_get_flex_aa_sets(self):
        amino_acids = ["A", "B", "C", "D"]

        result = PositionalMotifHelper.get_flex_aa_sets(amino_acids)
        expected = ["AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD", "ABCD"]

        self.assertListEqual(result, expected)

    # def test_identify_n_possible_motifs(self):
    #     np_sequences = np.asarray(['A' 'A' 'A', 'C' 'C' 'C', 'A' 'C' 'A', 'C' 'D' 'C']).view('U1').reshape(4, -1)
    #
    #     expected = {1: 7, 2: 10, 3: 4, 4: 0}
    #
    #     result = PositionalMotifHelper.identify_n_possible_motifs(np_sequences, 1, [1,2,3,4])
    #
    #     self.assertDictEqual(result, expected)
    #
    #     # problem: current approach is not looking at combinations of positions occurring at least once, just all motifs made up of individual positions!
