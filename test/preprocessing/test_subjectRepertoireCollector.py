import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
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

        reps = [Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAA", sequence_id="1")], result_path=path,
                                                metadata={"subject_id": "patient1"}),
                Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAC", sequence_id="2")], result_path=path,
                                                metadata={"subject_id": "patient1"}),
                Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAC", sequence_id="3")], result_path=path,
                                                metadata={"subject_id": "patient3"})]

        dataset = RepertoireDataset(repertoires=reps)

        dataset2 = SubjectRepertoireCollector().process_dataset(dataset, path / "result")

        self.assertEqual(2, len(dataset2.get_data()))
        self.assertEqual(3, len(dataset.get_data()))

        values = [2, 1]
        for index, rep in enumerate(dataset2.get_data()):
            self.assertEqual(values[index], len(rep.data))

        shutil.rmtree(path)
