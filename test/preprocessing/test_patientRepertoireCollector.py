import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.PatientRepertoireCollector import PatientRepertoireCollector
from source.util.PathBuilder import PathBuilder


class TestPatientRepertoireCollector(TestCase):
    def test_process(self):
        reps = [Repertoire(sequences=[ReceptorSequence("AAA")], identifier="patient1"),
                Repertoire(sequences=[ReceptorSequence("AAC")], identifier="patient1"),
                Repertoire(sequences=[ReceptorSequence("AAC")], identifier="patient2")]

        path = EnvironmentSettings.root_path + "test/tmp/patientrepertoirecollector/"
        PathBuilder.build(path)
        files = []
        for i, rep in enumerate(reps):
            files.append(path + "rep{}.pkl".format(len(reps)-1-i))
            with open(files[-1], "wb") as file:
                pickle.dump(rep, file)

        dataset = RepertoireDataset(filenames=files)

        dataset2 = PatientRepertoireCollector.process(dataset, {"result_path": path + "result/"})

        self.assertEqual(2, len(dataset2.get_filenames()))
        self.assertEqual(3, len(dataset.get_filenames()))

        values = [2, 1]
        for index, rep in enumerate(dataset2.get_data()):
            self.assertEqual(values[index], len(rep.sequences))

        shutil.rmtree(path)
