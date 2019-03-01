import pickle
import shutil
from unittest import TestCase

from gensim.models import Word2Vec

from source.data_model.dataset.Dataset import Dataset
from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from source.util.PathBuilder import PathBuilder


class TestSequenceModelCreator(TestCase):
    def test_create_model(self):
        test_path = "./w2v_test_tmp/"

        PathBuilder.build(test_path)

        sequence1 = ReceptorSequence("CASSVFA")
        sequence2 = ReceptorSequence("CASSCCC")

        metadata1 = RepertoireMetadata(custom_params={"T1D": "T1D"})
        rep1 = Repertoire([sequence1, sequence2], metadata1)
        file1 = test_path + "rep1.pkl"

        with open(file1, "wb") as file:
            pickle.dump(rep1, file)

        metadata2 = RepertoireMetadata(custom_params={"T1D": "CTL"})
        rep2 = Repertoire([sequence1], metadata2)
        file2 = test_path + "rep2.pkl"

        with open(file2, "wb") as file:
            pickle.dump(rep2, file)

        dataset = Dataset(filenames=[file1, file2])

        config_params = {
            "model": {
                "k": 2,
                "size": 16
            },
            "batch_size": 2
        }

        model_creator = SequenceModelCreator()
        model = model_creator.create_model(dataset=dataset, params=config_params, model_path=test_path + "model.model")

        self.assertTrue(isinstance(model, Word2Vec))
        self.assertTrue("CA" in model.wv.vocab)
        self.assertEqual(400, len(model.wv.vocab))

        shutil.rmtree(test_path)
