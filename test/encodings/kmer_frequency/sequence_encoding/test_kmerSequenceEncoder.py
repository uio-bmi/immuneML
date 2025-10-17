from unittest import TestCase

from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestKmerSequenceEncoder(TestCase):
    def test_encode_sequence(self):
        seq = ReceptorSequence(sequence_aa="CASSVFRTY")
        result = KmerSequenceEncoder.encode_sequence(seq, EncoderParams(model={"k": 3},
                                                                        label_config=LabelConfiguration(),
                                                                        result_path="", pool_size=4))

        self.assertTrue("CAS" in result)
        self.assertTrue("ASS" in result)
        self.assertTrue("SSV" in result)
        self.assertTrue("SVF" in result)
        self.assertTrue("VFR" in result)
        self.assertTrue("FRT" in result)
        self.assertTrue("RTY" in result)

        self.assertEqual(7, len(result))
        self.assertEqual(
            KmerSequenceEncoder.encode_sequence(ReceptorSequence(sequence_aa="AC"),
                                                EncoderParams(model={"k": 3}, label_config=LabelConfiguration(),
                                                              result_path="", pool_size=4)),
            None
        )
