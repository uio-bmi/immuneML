import shutil

import pytest

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting.ImplantingComputation import ImplantingComputation
from immuneML.util.PathBuilder import PathBuilder


@pytest.mark.skip(reason='simulation will come from ligo')
def test_implant_in_repertoire():
    path = EnvironmentSettings.tmp_test_path / "healthysequenceimplanting/"
    PathBuilder.remove_old_and_build(path)

    repertoire = Repertoire.build_from_sequence_objects([ReceptorSequence(sequence_aa="ACDFQ", sequence_id="1",
                                                                          metadata=SequenceMetadata(
                                                                              region_type='IMGT_CDR3')),
                                                         ReceptorSequence(sequence_aa="TGCDF", sequence_id="2",
                                                                          metadata=SequenceMetadata(
                                                                              region_type='IMGT_CDR3'))],
                                                        path=path, metadata={"subject_id": "1"})
    implanting = HealthySequenceImplanting(GappedMotifImplanting(), implanting_computation=ImplantingComputation.ROUND)
    signal = Signal("1", [Motif("m1", GappedKmerInstantiation(), "CCC")], implanting)

    repertoire2 = implanting.implant_in_repertoire(repertoire, 0.5, signal, path)

    new_sequences = [sequence.get_sequence() for sequence in repertoire2.sequences]
    assert "ACDFQ" in new_sequences or "TGCDF" in new_sequences
    assert any(["CCC" in sequence for sequence in new_sequences])

    shutil.rmtree(path)


@pytest.mark.skip(reason='simulation will come from ligo')
def test_implant_in_sequence():
    implanting = HealthySequenceImplanting(GappedMotifImplanting(), implanting_computation=ImplantingComputation.ROUND)
    signal = Signal("1", [Motif("m1", GappedKmerInstantiation(), "CCC")], implanting)
    sequence = ReceptorSequence(sequence_aa="ACDFQ", metadata=SequenceMetadata(region_type='IMGT_CDR3'))
    sequence2 = implanting.implant_in_sequence(sequence, signal)

    assert len(sequence.get_sequence()), len(sequence2.get_sequence())
    assert "CCC" in sequence2.get_sequence()
