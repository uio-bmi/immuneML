from unittest import TestCase

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder
from source.environment.LabelConfiguration import LabelConfiguration


class TestKmerSequenceEncoder(TestCase):
    def test_encode_sequence(self):
        seq = ReceptorSequence(amino_acid_sequence="CASSVFRTY")
        kmers = KmerSequenceEncoder.encode_sequence(seq, EncoderParams(model={"k": 3},
                                                                       label_configuration=LabelConfiguration(),
                                                                       result_path=""))

        self.assertTrue("CAS" in kmers)
        self.assertTrue("ASS" in kmers)
        self.assertTrue("SSV" in kmers)
        self.assertTrue("SVF" in kmers)
        self.assertTrue("VFR" in kmers)
        self.assertTrue("FRT" in kmers)
        self.assertTrue("RTY" in kmers)

        self.assertEqual(7, len(kmers))
        self.assertRaises(ValueError,
                          KmerSequenceEncoder.encode_sequence,
                          ReceptorSequence(amino_acid_sequence="AC"),
                          EncoderParams(model={"k": 3}, label_configuration=LabelConfiguration(), result_path=""))
