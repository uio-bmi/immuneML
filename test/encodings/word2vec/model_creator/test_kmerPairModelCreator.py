import shutil
from unittest import TestCase

from gensim.models import Word2Vec

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.encodings.word2vec.model_creator.KmerPairModelCreator import KmerPairModelCreator
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestKmerPairModelCreator(TestCase):
    def test_create_model(self):
        test_path = EnvironmentSettings.root_path + "test/tmp/w2v_test_tmp/"

        PathBuilder.build(test_path)

        sequence1 = ReceptorSequence("CASSVFA")
        sequence2 = ReceptorSequence("CASSCCC")

        metadata1 = {"T1D": "T1D"}
        rep1 = SequenceRepertoire.build_from_sequence_objects([sequence1, sequence2], test_path, "1", metadata1)

        metadata2 = {"T1D": "CTL"}
        rep2 = SequenceRepertoire.build_from_sequence_objects([sequence1], test_path, "2", metadata2)

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        model_creator = KmerPairModelCreator()
        model = model_creator.create_model(dataset=dataset, k=2, vector_size=16, batch_size=1, model_path=test_path+"model.model")

        self.assertTrue(isinstance(model, Word2Vec))
        self.assertTrue("CA" in model.wv.vocab)
        self.assertEqual(400, len(model.wv.vocab))

        shutil.rmtree(test_path)
