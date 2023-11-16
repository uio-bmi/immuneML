from unittest import TestCase

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType


class TestSequence(TestCase):
    def test_get_sequence(self):

        sequence = ReceptorSequence(sequence_aa="CAS",
                                    sequence="TGTGCTTCC")

        EnvironmentSettings.set_sequence_type(SequenceType.AMINO_ACID)

        self.assertEqual(sequence.get_sequence(), "CAS")
