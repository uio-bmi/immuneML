import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.Implanting import Implanting
from source.simulation.Simulation import Simulation
from source.simulation.SimulationState import SimulationState
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from source.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.workflows.steps.SignalImplanter import SignalImplanter


class TestSignalImplanter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):

        r = []

        path = EnvironmentSettings.root_path + "test/tmp/signalImplanter/"

        if not os.path.isdir(path):
            os.makedirs(path)

        sequences = [ReceptorSequence("ACDEFG", identifier="1"), ReceptorSequence("ACDEFG", identifier="2"),
                     ReceptorSequence("ACDEFG", identifier="3"), ReceptorSequence("ACDEFG", identifier="4")]

        for i in range(10):
            rep = Repertoire.build_from_sequence_objects(sequence_objects=sequences, path=path, metadata={})
            r.append(rep)

        dataset = RepertoireDataset(repertoires=r)

        m1 = Motif(identifier="m1", instantiation_strategy=GappedKmerInstantiation(), seed="CAS")
        m2 = Motif(identifier="m2", instantiation_strategy=GappedKmerInstantiation(), seed="CCC")
        s1 = Signal(identifier="s1", motifs=[m1], implanting_strategy=HealthySequenceImplanting(GappedMotifImplanting()))
        s2 = Signal(identifier="s2", motifs=[m1, m2],
                    implanting_strategy=HealthySequenceImplanting(GappedMotifImplanting()))

        simulation = Simulation([Implanting(dataset_implanting_rate=0.2, repertoire_implanting_rate=0.5, signals=[s1, s2]),
                                 Implanting(dataset_implanting_rate=0.2, repertoire_implanting_rate=0.5, signals=[s2])])

        input_params = SimulationState(dataset=dataset, result_path=path, simulation=simulation, signals=[s1, s2], batch_size=1)

        new_dataset = SignalImplanter.run(input_params)
        reps_with_s2 = sum([rep.metadata[f"signal_{s2.id}"] is True for rep in new_dataset.get_data(batch_size=10)])
        reps_with_s1 = sum([rep.metadata[f"signal_{s1.id}"] is True for rep in new_dataset.get_data(batch_size=10)])
        self.assertEqual(10, len(new_dataset.get_example_ids()))
        self.assertTrue(all([f"signal_{s1.id}" in rep.metadata.keys() for rep in new_dataset.get_data(batch_size=10)]))
        self.assertTrue(all([f"signal_{s2.id}" in rep.metadata.keys() for rep in new_dataset.get_data(batch_size=10)]))
        self.assertTrue(reps_with_s2 == 4)
        self.assertTrue(reps_with_s1 == 2)

        self.assertEqual(10, len(new_dataset.get_example_ids()))

        metadata_filenames = new_dataset.get_metadata(["filename"])["filename"]
        self.assertTrue(all([repertoire.data_filename in metadata_filenames for repertoire in new_dataset.repertoires]))

        shutil.rmtree(path)
