from unittest import TestCase

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.kmer_frequency.sequence_encoding.GappedKmerSequenceEncoder import GappedKmerSequenceEncoder


class TestGappedKmerSequenceEncoder(TestCase):
    def test_encode_sequence(self):
        sequence = ReceptorSequence("ABCDEFG", None, None)
        kmers = GappedKmerSequenceEncoder.encode_sequence(sequence, {"k_left": 3, "max_gap": 1})
        self.assertEqual({'ABC.EFG', 'ABCDEF', 'BCDEFG'}, set(kmers))

        with self.assertRaises(ValueError):
            GappedKmerSequenceEncoder.encode_sequence(sequence, {"k_left": 10, "max_gap": 1})

        sequence.amino_acid_sequence = "ABCDEFG"
        kmers = GappedKmerSequenceEncoder.encode_sequence(sequence, {"k_left": 3, "max_gap": 1})
        self.assertEqual({'ABC.EFG', 'ABCDEF', 'BCDEFG'}, set(kmers))

        with self.assertRaises(ValueError):
            GappedKmerSequenceEncoder.encode_sequence(sequence, {"k_left": 10, "max_gap": 1})

        sequence.amino_acid_sequence = "ABCDEFG"
        kmers = GappedKmerSequenceEncoder.encode_sequence(sequence, {"k_left": 2, "max_gap": 1, "min_gap": 1, "k_right": 3})
        self.assertEqual({'AB.DEF', 'BC.EFG'}, set(kmers))

