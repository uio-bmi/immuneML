from unittest import TestCase

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.IdentitySequenceEncoder import IdentitySequenceEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestIdentitySequenceEncoder(TestCase):
    def test_encode_sequence(self):
        sequence = ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="OUT"))
        enc = IdentitySequenceEncoder()
        self.assertEqual(enc.encode_sequence(sequence, EncoderParams(model={},
                                                                     label_config=LabelConfiguration(),
                                                                     result_path="")),
                         ["AAA"])

        sequence = ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="STOP"))
        enc = IdentitySequenceEncoder()
        self.assertEqual(enc.encode_sequence(sequence, EncoderParams(model={},
                                                                     label_config=LabelConfiguration(),
                                                                     result_path="")),
                         ["AAA"])

        sequence = ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(frame_type="IN"))
        enc = IdentitySequenceEncoder()
        self.assertEqual(["AAA"],
                         enc.encode_sequence(sequence, EncoderParams(model={},
                                                                     label_config=LabelConfiguration(),
                                                                     result_path="")))
