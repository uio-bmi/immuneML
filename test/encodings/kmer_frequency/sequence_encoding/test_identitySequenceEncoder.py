from unittest import TestCase

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.encodings.kmer_frequency.sequence_encoding.IdentitySequenceEncoder import IdentitySequenceEncoder


class TestIdentitySequenceEncoder(TestCase):
    def test_encode_sequence(self):
        sequence = ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="Out"))
        enc = IdentitySequenceEncoder()
        self.assertIsNone(enc.encode_sequence(sequence, {}))

        sequence = ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="In"))
        enc = IdentitySequenceEncoder()
        self.assertEqual(["AAA"], enc.encode_sequence(sequence, {}))
