import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.LabelType import LabelType
from source.util.PathBuilder import PathBuilder


class TestWord2VecEncoder(TestCase):
    def test_encode(self):

        test_path = EnvironmentSettings.root_path + "test/tmp/w2v/"

        PathBuilder.build(test_path)

        sequence1 = ReceptorSequence("CASSVFA")
        sequence2 = ReceptorSequence("CASSCCC")

        metadata1 = RepertoireMetadata()
        metadata1.custom_params = {"T1D": "T1D"}
        rep1 = Repertoire([sequence1, sequence2], metadata1)
        file1 = test_path + "rep1.pkl"

        with open(file1, "wb") as file:
            pickle.dump(rep1, file)

        metadata2 = RepertoireMetadata()
        metadata2.custom_params = {"T1D": "CTL"}
        rep2 = Repertoire([sequence1], metadata2)
        file2 = test_path + "rep2.pkl"

        with open(file2, "wb") as file:
            pickle.dump(rep2, file)

        dataset = Dataset(filenames=[file1, file2])

        label_configuration = LabelConfiguration()
        label_configuration.add_label("T1D", ["T1D", "CTL"], LabelType.CLASSIFICATION)

        config_params = EncoderParams(
            model={
                "k": 3,
                "model_creator": ModelType.SEQUENCE,
                "size": 16
            },
            batch_size=1,
            learn_model=True,
            result_path=test_path,
            label_configuration=label_configuration,
            model_path=test_path,
            scaler_path=test_path,
            filename="dataset.pkl"
        )

        encoded_dataset = Word2VecEncoder.encode(dataset=dataset, params=config_params)

        self.assertIsNotNone(encoded_dataset.encoded_data)
        self.assertTrue("repertoires" in encoded_dataset.encoded_data)
        self.assertTrue(encoded_dataset.encoded_data["repertoires"].shape[0] == 2)
        self.assertTrue(encoded_dataset.encoded_data["repertoires"].shape[1] == 16)
        self.assertTrue("labels" in encoded_dataset.encoded_data)
        self.assertTrue(len(encoded_dataset.encoded_data["labels"][0]) == 2)
        self.assertTrue(encoded_dataset.encoded_data["labels"][0][0] == "T1D")

        shutil.rmtree(test_path)
