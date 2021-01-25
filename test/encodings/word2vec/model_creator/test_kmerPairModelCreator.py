import shutil
from unittest import TestCase

from gensim.models import Word2Vec

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.word2vec.model_creator.KmerPairModelCreator import KmerPairModelCreator
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestKmerPairModelCreator(TestCase):
    def test_create_model(self):
        test_path = EnvironmentSettings.root_path / "test/tmp/w2v_test_tmp/"

        PathBuilder.build(test_path)

        sequence1 = ReceptorSequence("CASSVFA")
        sequence2 = ReceptorSequence("CASSCCC")

        metadata1 = {"T1D": "T1D", "subject_id": "1"}
        rep1 = Repertoire.build_from_sequence_objects([sequence1, sequence2], test_path, metadata1)

        metadata2 = {"T1D": "CTL", "subject_id": "2"}
        rep2 = Repertoire.build_from_sequence_objects([sequence1], test_path, metadata2)

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        model_creator = KmerPairModelCreator()
        model = model_creator.create_model(dataset=dataset, k=2, vector_size=16, batch_size=1, model_path=test_path/"model.model")

        self.assertTrue(isinstance(model, Word2Vec))
        self.assertTrue("CA" in model.wv.vocab)
        self.assertEqual(400, len(model.wv.vocab))

        shutil.rmtree(test_path)
