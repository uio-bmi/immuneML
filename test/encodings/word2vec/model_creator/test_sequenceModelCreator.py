import shutil
from unittest import TestCase

from gensim.models import Word2Vec

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceModelCreator(TestCase):
    def test_create_model(self):
        test_path = EnvironmentSettings.tmp_test_path / "w2vseqmc/"

        PathBuilder.remove_old_and_build(test_path)

        sequence1 = ReceptorSequence(sequence_aa="CASSVFA", sequence_id="1")
        sequence2 = ReceptorSequence(sequence_aa="CASSCCC", sequence_id="2")

        metadata1 = {"T1D": "T1D", "subject_id": "1"}
        rep1 = Repertoire.build_from_sequences([sequence1, sequence2], metadata=metadata1, result_path=test_path)

        metadata2 = {"T1D": "CTL", "subject_id": "2"}
        rep2 = Repertoire.build_from_sequences([sequence1], metadata=metadata2, result_path=test_path)

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        model_creator = SequenceModelCreator(epochs=10, window=5)
        model = model_creator.create_model(dataset=dataset, k=2, vector_size=16, batch_size=2, model_path=test_path / "model.model",
                                           sequence_type=SequenceType.AMINO_ACID)

        self.assertTrue(isinstance(model, Word2Vec))
        self.assertTrue("CA" in model.wv.key_to_index)
        self.assertEqual(400, len(model.wv.key_to_index))

        shutil.rmtree(test_path)
