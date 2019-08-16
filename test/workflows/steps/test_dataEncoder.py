import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.metadata.Sample import Sample
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams


class TestDataEncoder(TestCase):
    def test_run(self):
        encoder = Word2VecEncoder()
        path = EnvironmentSettings.root_path + "test/tmp/dataencoder/"
        PathBuilder.build(path)

        rep1 = Repertoire(sequences=[ReceptorSequence("AAA")],
                          metadata=RepertoireMetadata(Sample(1), custom_params={"l1": 1, "l2": 2}))
        with open(path + "rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)

        rep2 = Repertoire(sequences=[ReceptorSequence("ATA")],
                          metadata=RepertoireMetadata(Sample(2), custom_params={"l1": 0, "l2": 3}))
        with open(path + "rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = Dataset(filenames=[path + "rep1.pkl", path + "rep2.pkl"])

        res = DataEncoder.run(DataEncoderParams(
            dataset=dataset,
            encoder=encoder,
            encoder_params=EncoderParams(
                model={
                    "k": 3,
                    "model_creator": ModelType.SEQUENCE,
                    "size": 6
                },
                batch_size=2,
                label_configuration=lc,
                result_path=path,
                filename="dataset.csv"
            )
        ))

        self.assertTrue(isinstance(res, Dataset))
        self.assertTrue(res.encoded_data.repertoires.shape[0] == 2)

        shutil.rmtree(path)
