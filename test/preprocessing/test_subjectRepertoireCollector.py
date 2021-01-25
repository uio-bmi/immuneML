import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.SubjectRepertoireCollector import SubjectRepertoireCollector
from immuneML.util.PathBuilder import PathBuilder


class TestSubjectRepertoireCollector(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = EnvironmentSettings.root_path / "test/tmp/subject_rep_collector"
        PathBuilder.build(path)

        reps = [Repertoire.build_from_sequence_objects([ReceptorSequence("AAA", identifier="1")], path=path,
                                                       metadata={"subject_id": "patient1"}),
                Repertoire.build_from_sequence_objects([ReceptorSequence("AAC", identifier="2")], path=path,
                                                       metadata={"subject_id": "patient1"}),
                Repertoire.build_from_sequence_objects([ReceptorSequence("AAC", identifier="3")], path=path,
                                                       metadata={"subject_id": "patient3"})]

        dataset = RepertoireDataset(repertoires=reps)

        dataset2 = SubjectRepertoireCollector.process(dataset, {"result_path": path / "result"})

        self.assertEqual(2, len(dataset2.get_data()))
        self.assertEqual(3, len(dataset.get_data()))

        values = [2, 1]
        for index, rep in enumerate(dataset2.get_data()):
            self.assertEqual(values[index], len(rep.sequences))

        shutil.rmtree(path)
