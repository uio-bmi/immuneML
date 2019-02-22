from unittest import TestCase

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.IdentitySequenceEncoder import IdentitySequenceEncoder
from source.environment.LabelConfiguration import LabelConfiguration


class TestIdentitySequenceEncoder(TestCase):
    def test_encode_sequence(self):
        sequence = ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="Out"))
        enc = IdentitySequenceEncoder()
        self.assertIsNone(enc.encode_sequence(sequence, EncoderParams(model={},
                                                                      label_configuration=LabelConfiguration(),
                                                                      result_path="")))

        sequence = ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="In"))
        enc = IdentitySequenceEncoder()
        self.assertEqual(["AAA"], enc.encode_sequence(sequence, EncoderParams(model={},
                                                                              label_configuration=LabelConfiguration(),
                                                                              result_path="")))
