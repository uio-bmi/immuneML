from unittest import TestCase

from source.data_model.sequence.Sequence import Sequence
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.SequenceType import SequenceType


class TestSequence(TestCase):
    def test_get_sequence(self):

        sequence = Sequence(amino_acid_sequence="CAS",
                            nucleotide_sequence="TGTGCTTCC")

        EnvironmentSettings.set_sequence_type(SequenceType.AMINO_ACID)

        self.assertEqual(sequence.get_sequence(), "CAS")

        EnvironmentSettings.set_sequence_type(SequenceType.NUCLEOTIDE)

        self.assertEqual(sequence.get_sequence(), "TGTGCTTCC")
