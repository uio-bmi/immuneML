from unittest import TestCase

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting


class TestGappedMotifImplanting(TestCase):
    def test_implant(self):

        strategy = GappedMotifImplanting()
        motif_instance = MotifInstance("CC/T", 2)
        sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                     metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                         "motif_id": "1",
                                                                                         "motif_instance": motif_instance})

        self.assertTrue(sequence.get_sequence().find("CCAAT") > -1)
        self.assertEqual(10, len(sequence.get_sequence()))

        sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                     metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                         "motif_id": "1",
                                                                                         "motif_instance": motif_instance},
                                    sequence_position_weights={105: 0.8, 106: 0.2})

        self.assertTrue(-1 < sequence.get_sequence().find("CCAAT") < 2)
        self.assertEqual(10, len(sequence.get_sequence()))

        motif_instance = MotifInstance("CCT", 0)
        sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                     metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                         "motif_id": "1",
                                                                                         "motif_instance": motif_instance},
                                    sequence_position_weights={105: 0.8, 106: 0.2})

        self.assertTrue(-1 < sequence.get_sequence().find("CCT") < 2)
        self.assertEqual(10, len(sequence.get_sequence()))

        motif_instance = MotifInstance("C/T", 0)
        sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                     metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                         "motif_id": "1",
                                                                                         "motif_instance": motif_instance},
                                    sequence_position_weights={105: 0.8, 106: 0.2})

        self.assertTrue(-1 < sequence.get_sequence().find("CT") < 2)
        self.assertTrue("/" not in sequence.get_sequence())
