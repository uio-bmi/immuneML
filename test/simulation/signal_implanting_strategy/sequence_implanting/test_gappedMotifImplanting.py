from unittest import TestCase

from source.data_model.sequence.Sequence import Sequence
from source.simulation.implants.MotifInstance import MotifInstance
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting


class TestGappedMotifImplanting(TestCase):
    def test_implant(self):

        strategy = GappedMotifImplanting()
        motif_instance = MotifInstance("CC/T", 2)
        sequence = strategy.implant(Sequence(amino_acid_sequence="AAAAAAAAAA"), {"signal_id": "1",
                                                                                 "motif_id": "1",
                                                                                 "motif_instance": motif_instance})

        print(sequence.get_sequence())
        self.assertTrue(sequence.get_sequence().find("CCAAT") > -1)
        self.assertEqual(10, len(sequence.get_sequence()))

        sequence = strategy.implant(Sequence(amino_acid_sequence="AAAAAAAAAA"), {"signal_id": "1",
                                                                                 "motif_id": "1",
                                                                                 "motif_instance": motif_instance},
                                    sequence_position_weights={105: 0.8, 106: 0.2})

        print(sequence.get_sequence())
        self.assertTrue(-1 < sequence.get_sequence().find("CCAAT") < 2)
        self.assertEqual(10, len(sequence.get_sequence()))

        motif_instance = MotifInstance("CCT", 0)
        sequence = strategy.implant(Sequence(amino_acid_sequence="AAAAAAAAAA"), {"signal_id": "1",
                                                                                 "motif_id": "1",
                                                                                 "motif_instance": motif_instance},
                                    sequence_position_weights={105: 0.8, 106: 0.2})

        print(sequence.get_sequence())
        self.assertTrue(-1 < sequence.get_sequence().find("CCT") < 2)
        self.assertEqual(10, len(sequence.get_sequence()))
