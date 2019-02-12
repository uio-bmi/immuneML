from unittest import TestCase

from source.data_model.sequence.Sequence import Sequence
from source.data_model.sequence.SequenceMetadata import SequenceMetadata
from source.encodings.kmer_frequency.sequence_encoding.IdentitySequenceEncoder import IdentitySequenceEncoder


class TestIdentitySequenceEncoder(TestCase):
    def test_encode_sequence(self):
        sequence = Sequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="Out"))
        enc = IdentitySequenceEncoder()
        self.assertIsNone(enc.encode_sequence(sequence, {}))

        sequence = Sequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="In"))
        enc = IdentitySequenceEncoder()
        self.assertEqual(["AAA"], enc.encode_sequence(sequence, {}))
