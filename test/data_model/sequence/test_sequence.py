from unittest import TestCase

from source.data_model.receptor.receptor_sequence import ReceptorSequence
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.SequenceType import SequenceType


class TestSequence(TestCase):
    def test_get_sequence(self):

        sequence = ReceptorSequence(amino_acid_sequence="CAS",
                                    nucleotide_sequence="TGTGCTTCC")

        EnvironmentSettings.set_sequence_type(SequenceType.AMINO_ACID)

        self.assertEqual(sequence.get_sequence(), "CAS")
