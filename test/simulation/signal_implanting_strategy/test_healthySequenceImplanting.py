from unittest import TestCase

from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.sequence.Sequence import Sequence
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.IdentityMotifInstantiation import IdentityMotifInstantiation
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting


class TestHealthySequenceImplanting(TestCase):
    def test_implant_in_repertoire(self):
        repertoire = Repertoire([Sequence(amino_acid_sequence="ACDFQ"), Sequence(amino_acid_sequence="TGCDF")])
        implanting = HealthySequenceImplanting(GappedMotifImplanting())
        signal = Signal(1, [Motif("m1", IdentityMotifInstantiation(), "CCC")], implanting)

        repertoire2 = implanting.implant_in_repertoire(repertoire, 0.5, signal)

        new_sequences = [sequence.get_sequence() for sequence in repertoire2.sequences]
        self.assertTrue("ACDFQ" in new_sequences or "TGCDF" in new_sequences)
        self.assertTrue(any(["CCC" in sequence for sequence in new_sequences]))

    def test_implant_in_sequence(self):
        implanting = HealthySequenceImplanting(GappedMotifImplanting())
        signal = Signal(1, [Motif("m1", IdentityMotifInstantiation(), "CCC")], implanting)
        sequence = Sequence(amino_acid_sequence="ACDFQ")
        sequence2 = implanting.implant_in_sequence(sequence, signal)

        self.assertEqual(len(sequence.get_sequence()), len(sequence2.get_sequence()))
        self.assertTrue("CCC" in sequence2.get_sequence())
