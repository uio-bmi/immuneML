import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.PatientRepertoireCollector import PatientRepertoireCollector
from source.util.PathBuilder import PathBuilder


class TestPatientRepertoireCollector(TestCase):
    def test_process(self):
        path = EnvironmentSettings.root_path + "test/tmp/patientrepertoirecollector/"
        PathBuilder.build(path)

        reps = [SequenceRepertoire.build_from_sequence_objects([ReceptorSequence("AAA", identifier="1")],
                                                               identifier="patient1", path=path, metadata={}),
                SequenceRepertoire.build_from_sequence_objects([ReceptorSequence("AAC", identifier="2")],
                                                               identifier="patient1", path=path, metadata={}),
                SequenceRepertoire.build_from_sequence_objects([ReceptorSequence("AAC", identifier="3")],
                                                               identifier="patient3", path=path, metadata={})]

        dataset = RepertoireDataset(repertoires=reps)

        dataset2 = PatientRepertoireCollector.process(dataset, {"result_path": path + "result/"})

        self.assertEqual(2, len(dataset2.get_data()))
        self.assertEqual(3, len(dataset.get_data()))

        values = [2, 1]
        for index, rep in enumerate(dataset2.get_data()):
            self.assertEqual(values[index], len(rep.sequences))

        shutil.rmtree(path)
