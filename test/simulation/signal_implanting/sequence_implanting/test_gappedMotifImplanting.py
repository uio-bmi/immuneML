import pytest

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting


@pytest.mark.skip(reason='simulation will come from ligo')
def test_implant():
    strategy = GappedMotifImplanting()
    motif_instance = MotifInstance("CC/T", 2)
    sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                 metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                                       "motif_id": "1",
                                                                                                       "motif_instance": motif_instance})

    assert sequence.get_sequence().find("CCAAT") > -1
    assert 10 == len(sequence.get_sequence())

    sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                 metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                                       "motif_id": "1",
                                                                                                       "motif_instance": motif_instance},
                                sequence_position_weights={105: 0.8, 106: 0.2})

    assert -1 < sequence.get_sequence().find("CCAAT") < 2
    assert 10 == len(sequence.get_sequence())

    motif_instance = MotifInstance("CCT", 0)
    sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                 metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                                       "motif_id": "1",
                                                                                                       "motif_instance": motif_instance},
                                sequence_position_weights={105: 0.8, 106: 0.2})

    assert -1 < sequence.get_sequence().find("CCT") < 2
    assert 10 == len(sequence.get_sequence())

    motif_instance = MotifInstance("C/T", 0)
    sequence = strategy.implant(ReceptorSequence(sequence_aa="AAAAAAAAAA",
                                                 metadata=SequenceMetadata(region_type="IMGT_CDR3")), {"signal_id": "1",
                                                                                                       "motif_id": "1",
                                                                                                       "motif_instance": motif_instance},
                                sequence_position_weights={105: 0.8, 106: 0.2})

    assert -1 < sequence.get_sequence().find("CT") < 2
    assert "/" not in sequence.get_sequence()
