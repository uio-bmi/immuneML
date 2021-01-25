import shutil
from unittest import TestCase

from gensim.models import Word2Vec

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceModelCreator(TestCase):
    def test_create_model(self):
        test_path = EnvironmentSettings.root_path / "test/tmp/w2vseqmc/"

        PathBuilder.build(test_path)

        sequence1 = ReceptorSequence("CASSVFA", identifier="1")
        sequence2 = ReceptorSequence("CASSCCC", identifier="2")

        metadata1 = {"T1D": "T1D", "subject_id": "1"}
        rep1 = Repertoire.build_from_sequence_objects([sequence1, sequence2], metadata=metadata1, path=test_path)

        metadata2 = {"T1D": "CTL", "subject_id": "2"}
        rep2 = Repertoire.build_from_sequence_objects([sequence1], metadata=metadata2, path=test_path)

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        model_creator = SequenceModelCreator()
        model = model_creator.create_model(dataset=dataset, k=2, vector_size=16, batch_size=2, model_path=test_path / "model.model")

        self.assertTrue(isinstance(model, Word2Vec))
        self.assertTrue("CA" in model.wv.vocab)
        self.assertEqual(400, len(model.wv.vocab))

        shutil.rmtree(test_path)
