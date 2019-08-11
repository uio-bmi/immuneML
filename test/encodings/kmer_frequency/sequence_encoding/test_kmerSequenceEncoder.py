from unittest import TestCase

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder
from source.environment.LabelConfiguration import LabelConfiguration


class TestKmerSequenceEncoder(TestCase):
    def test_encode_sequence(self):
        seq = ReceptorSequence(amino_acid_sequence="CASSVFRTY")
        result = KmerSequenceEncoder.encode_sequence(seq, EncoderParams(model={"k": 3},
                                                                       label_configuration=LabelConfiguration(),
                                                                       result_path=""))

        self.assertTrue("CAS" in result)
        self.assertTrue("ASS" in result)
        self.assertTrue("SSV" in result)
        self.assertTrue("SVF" in result)
        self.assertTrue("VFR" in result)
        self.assertTrue("FRT" in result)
        self.assertTrue("RTY" in result)

        self.assertEqual(7, len(result))
        self.assertEqual(
            KmerSequenceEncoder.encode_sequence(
                ReceptorSequence(amino_acid_sequence="AC"),
                EncoderParams(model={"k": 3}, label_configuration=LabelConfiguration(), result_path="")
            ),
            None
        )
