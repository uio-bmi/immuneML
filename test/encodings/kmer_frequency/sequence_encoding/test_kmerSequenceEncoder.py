from unittest import TestCase

from source.data_model.sequence.Sequence import Sequence
from source.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder


class TestKmerSequenceEncoder(TestCase):
    def test_encode_sequence(self):
        seq = Sequence(amino_acid_sequence="CASSVFRTY")
        kmers = KmerSequenceEncoder.encode_sequence(seq, {
            "k": 3
        })

        self.assertTrue("CAS" in kmers)
        self.assertTrue("ASS" in kmers)
        self.assertTrue("SSV" in kmers)
        self.assertTrue("SVF" in kmers)
        self.assertTrue("VFR" in kmers)
        self.assertTrue("FRT" in kmers)
        self.assertTrue("RTY" in kmers)

        self.assertEqual(7, len(kmers))
        self.assertRaises(ValueError, KmerSequenceEncoder.encode_sequence, Sequence(amino_acid_sequence="AC"), {"k": 3})
