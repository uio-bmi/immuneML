from unittest import TestCase

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.GappedKmerSequenceEncoder import GappedKmerSequenceEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestGappedKmerSequenceEncoder(TestCase):
    def test_encode_sequence(self):
        sequence = ReceptorSequence(sequence_aa="ABCDEFG", sequence=None, sequence_id=None)
        result = GappedKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 3, "max_gap": 1},
                                                                                   label_config=LabelConfiguration(),
                                                                                   result_path="",
                                                                                   region_type=RegionType.IMGT_CDR3))
        self.assertEqual({'ABC.EFG', 'ABCDEF', 'BCDEFG'}, set(result))
        result = GappedKmerSequenceEncoder.get_feature_names(EncoderParams(model={"k_left": 3, "max_gap": 1},
                                                                           label_config=LabelConfiguration(),
                                                                           result_path="",
                                                                           region_type=RegionType.IMGT_CDR3))
        self.assertEqual({'sequence'}, set(result))

        self.assertEqual(GappedKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 10, "max_gap": 1},
                                                                                           label_config=LabelConfiguration(),
                                                                                           result_path="")),
                         None)

        sequence.sequence_aa = "ABCDEFG"
        result = GappedKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 3, "max_gap": 1},
                                                                                   label_config=LabelConfiguration(),
                                                                                   result_path=""))
        self.assertEqual({'ABC.EFG', 'ABCDEF', 'BCDEFG'}, set(result))
        result = GappedKmerSequenceEncoder.get_feature_names(EncoderParams(model={"k_left": 3, "max_gap": 1},
                                                                           label_config=LabelConfiguration(),
                                                                           result_path=""))
        self.assertEqual({'sequence'}, set(result))

        self.assertEqual(GappedKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 10, "max_gap": 1},
                                                                                           label_config=LabelConfiguration(),
                                                                                           result_path="")),
                         None)

        sequence.sequence_aa = "ABCDEFG"
        result = GappedKmerSequenceEncoder.encode_sequence(sequence,
                                                           EncoderParams(model={"k_left": 2,
                                                                                "max_gap": 1,
                                                                                "min_gap": 1,
                                                                                "k_right": 3},
                                                                         label_config=LabelConfiguration(),
                                                                         result_path=""))
        self.assertEqual({'AB.DEF', 'BC.EFG'}, set(result))
        result = GappedKmerSequenceEncoder.get_feature_names(EncoderParams(model={"k_left": 2,
                                                                                  "max_gap": 1,
                                                                                  "min_gap": 1,
                                                                                  "k_right": 3},
                                                                           label_config=LabelConfiguration(),
                                                                           result_path=""))
        self.assertEqual({'sequence'}, set(result))
