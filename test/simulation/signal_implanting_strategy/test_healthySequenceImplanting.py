from unittest import TestCase

from source.data_model.receptor.receptor_sequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.IdentityInstantiation import IdentityInstantiation
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting


class TestHealthySequenceImplanting(TestCase):
    def test_implant_in_repertoire(self):
        repertoire = Repertoire([ReceptorSequence(amino_acid_sequence="ACDFQ"), ReceptorSequence(amino_acid_sequence="TGCDF")])
        implanting = HealthySequenceImplanting(GappedMotifImplanting())
        signal = Signal(1, [Motif("m1", IdentityInstantiation(), "CCC")], implanting)

        repertoire2 = implanting.implant_in_repertoire(repertoire, 0.5, signal)

        new_sequences = [sequence.get_sequence() for sequence in repertoire2.sequences]
        self.assertTrue("ACDFQ" in new_sequences or "TGCDF" in new_sequences)
        self.assertTrue(any(["CCC" in sequence for sequence in new_sequences]))

    def test_implant_in_sequence(self):
        implanting = HealthySequenceImplanting(GappedMotifImplanting())
        signal = Signal(1, [Motif("m1", IdentityInstantiation(), "CCC")], implanting)
        sequence = ReceptorSequence(amino_acid_sequence="ACDFQ")
        sequence2 = implanting.implant_in_sequence(sequence, signal)

        self.assertEqual(len(sequence.get_sequence()), len(sequence2.get_sequence()))
        self.assertTrue("CCC" in sequence2.get_sequence())
