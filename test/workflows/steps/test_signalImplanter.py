import os
import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.Simulation import Simulation
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.IdentityInstantiation import IdentityInstantiation
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from source.workflows.steps.SignalImplanter import SignalImplanter
from source.workflows.steps.SignalImplanterParams import SignalImplanterParams


class TestSignalImplanter(TestCase):
    def test_run(self):

        r = []

        path = EnvironmentSettings.root_path + "test/tmp/signalImplanter/"

        if not os.path.isdir(path):
            os.makedirs(path)

        for i in range(10):
            rep = Repertoire(sequences=[ReceptorSequence("ACDEFG"), ReceptorSequence("ACDEFG"), ReceptorSequence("ACDEFG"), ReceptorSequence("ACDEFG")])
            filename = path + "rep" + str(i+1) + ".pkl"
            with open(filename, "wb") as file:
                pickle.dump(rep, file)
            r.append(filename)

        dataset = Dataset(filenames=r)

        m1 = Motif(identifier="m1", instantiation_strategy=IdentityInstantiation(), seed="CAS")
        m2 = Motif(identifier="m2", instantiation_strategy=IdentityInstantiation(), seed="CCC")
        s1 = Signal(identifier="s1", motifs=[m1], implanting_strategy=HealthySequenceImplanting(GappedMotifImplanting()))
        s2 = Signal(identifier="s2", motifs=[m1, m2],
                    implanting_strategy=HealthySequenceImplanting(GappedMotifImplanting()))

        simulations = [Simulation(dataset_implanting_rate=0.2, repertoire_implanting_rate=0.5, signals=[s1, s2]),
                       Simulation(dataset_implanting_rate=0.2, repertoire_implanting_rate=0.5, signals=[s2])]

        input_params = SignalImplanterParams(dataset=dataset, result_path=path, simulations=simulations, signals=[s1, s2], batch_size=1)

        new_dataset = SignalImplanter.run(input_params)
        reps_with_s2 = sum([rep.metadata.custom_params[s2.id] is True for rep in new_dataset.get_data(batch_size=10)])
        reps_with_s1 = sum([rep.metadata.custom_params[s1.id] is True for rep in new_dataset.get_data(batch_size=10)])
        self.assertEqual(10, len(new_dataset.get_filenames()))
        self.assertTrue(all([s1.id in rep.metadata.custom_params.keys() for rep in new_dataset.get_data(batch_size=10)]))
        self.assertTrue(all([s2.id in rep.metadata.custom_params.keys() for rep in new_dataset.get_data(batch_size=10)]))
        self.assertTrue(reps_with_s2 == 4)
        self.assertTrue(reps_with_s1 == 2)

        shutil.rmtree(path)
