import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.DatasetParams import DatasetParams
from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataEncoder import DataEncoder


class TestDataEncoder(TestCase):
    def test_run(self):
        encoder = Word2VecEncoder()

        PathBuilder.build("./tmp/")

        rep1 = Repertoire(sequences=[ReceptorSequence("AAA")],
                          metadata=RepertoireMetadata(Sample(1, custom_params={"l1": 1, "l2": 2})))
        with open("./tmp/rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)

        rep2 = Repertoire(sequences=[ReceptorSequence("ATA")],
                          metadata=RepertoireMetadata(Sample(2, custom_params={"l1": 0, "l2": 3})))
        with open("./tmp/rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = Dataset(filenames=["./tmp/rep1.pkl", "./tmp/rep2.pkl"], dataset_params=DatasetParams())

        res = DataEncoder.run({
            "dataset": dataset,
            "encoder": encoder,
            "encoder_params": EncoderParams(
                model={
                    "k": 3,
                    "model_creator": ModelType.SEQUENCE,
                    "size": 6
                },
                batch_size=2,
                label_configuration=lc,
                model_path="./tmp/",
                scaler_path="./tmp/",
                result_path="./tmp/"
            )
        })

        self.assertTrue(isinstance(res, Dataset))
        self.assertTrue(res.encoded_data["repertoires"].shape[0] == 2)

        shutil.rmtree("./tmp/")
