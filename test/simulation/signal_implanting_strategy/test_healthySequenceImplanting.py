import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from source.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation
from source.util.PathBuilder import PathBuilder


class TestHealthySequenceImplanting(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_implant_in_repertoire(self):
        path = EnvironmentSettings.tmp_test_path + "healthysequenceimplanting/"
        PathBuilder.build(path)

        repertoire = Repertoire.build_from_sequence_objects([ReceptorSequence(amino_acid_sequence="ACDFQ", identifier="1"),
                                                             ReceptorSequence(amino_acid_sequence="TGCDF", identifier="2")],
                                                            path=path, metadata={"subject_id": "1"})
        implanting = HealthySequenceImplanting(GappedMotifImplanting(), implanting_computation=ImplantingComputation.ROUND)
        signal = Signal("1", [Motif("m1", GappedKmerInstantiation(), "CCC")], implanting)

        repertoire2 = implanting.implant_in_repertoire(repertoire, 0.5, signal, path)

        new_sequences = [sequence.get_sequence() for sequence in repertoire2.sequences]
        self.assertTrue("ACDFQ" in new_sequences or "TGCDF" in new_sequences)
        self.assertTrue(any(["CCC" in sequence for sequence in new_sequences]))

        shutil.rmtree(path)

    def test_implant_in_sequence(self):
        implanting = HealthySequenceImplanting(GappedMotifImplanting(), implanting_computation=ImplantingComputation.ROUND)
        signal = Signal("1", [Motif("m1", GappedKmerInstantiation(), "CCC")], implanting)
        sequence = ReceptorSequence(amino_acid_sequence="ACDFQ")
        sequence2 = implanting.implant_in_sequence(sequence, signal)

        self.assertEqual(len(sequence.get_sequence()), len(sequence2.get_sequence()))
        self.assertTrue("CCC" in sequence2.get_sequence())
