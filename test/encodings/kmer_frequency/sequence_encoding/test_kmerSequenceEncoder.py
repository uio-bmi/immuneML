from unittest import TestCase

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingResult import SequenceEncodingResult
from source.environment.LabelConfiguration import LabelConfiguration


class TestKmerSequenceEncoder(TestCase):
    def test_encode_sequence(self):
        seq = ReceptorSequence(amino_acid_sequence="CASSVFRTY")
        result = KmerSequenceEncoder.encode_sequence(seq, EncoderParams(model={"k": 3},
                                                                       label_configuration=LabelConfiguration(),
                                                                       result_path=""))

        self.assertTrue("CAS" in result.features)
        self.assertTrue("ASS" in result.features)
        self.assertTrue("SSV" in result.features)
        self.assertTrue("SVF" in result.features)
        self.assertTrue("VFR" in result.features)
        self.assertTrue("FRT" in result.features)
        self.assertTrue("RTY" in result.features)

        self.assertEqual(7, len(result.features))
        self.assertEqual(
            KmerSequenceEncoder.encode_sequence(
                ReceptorSequence(amino_acid_sequence="AC"),
                EncoderParams(model={"k": 3}, label_configuration=LabelConfiguration(), result_path="")
            ),
            SequenceEncodingResult(None, None)
        )
