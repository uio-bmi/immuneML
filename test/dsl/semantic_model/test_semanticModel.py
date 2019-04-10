import glob
import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.dsl.semantic_model.SemanticModel import SemanticModel
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.util.PathBuilder import PathBuilder


class TestSemanticModel(TestCase):

    def test_add_ml_connection(self):
        model = SemanticModel()

        with self.assertRaises(AssertionError):
            model.add_ml_connection("m1", {})

    def test_add_report_connection(self):
        model = SemanticModel()

        with self.assertRaises(AssertionError):
            model.add_report_connection("r1", {"random_params": "e1"})

    def prepare_dataset(self, path) -> Dataset:
        rep1 = Repertoire(sequences=[ReceptorSequence("AAA"), ReceptorSequence("ATA"), ReceptorSequence("ATA")],
                          metadata=RepertoireMetadata(custom_params={"l1": 1, "l2": 2}))
        with open(path + "rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)

        rep2 = Repertoire(sequences=[ReceptorSequence("ATA"), ReceptorSequence("TAA"), ReceptorSequence("AAC")],
                          metadata=RepertoireMetadata(custom_params={"l1": 0, "l2": 3}))
        with open(path + "rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        rep3 = Repertoire(sequences=[ReceptorSequence("ATA"), ReceptorSequence("TAA"), ReceptorSequence("AAC")],
                          metadata=RepertoireMetadata(custom_params={"l1": 1, "l2": 3}))
        with open(path + "rep3.pkl", "wb") as file:
            pickle.dump(rep3, file)

        rep4 = Repertoire(sequences=[ReceptorSequence("ATA"), ReceptorSequence("TAA"), ReceptorSequence("AAC")],
                          metadata=RepertoireMetadata(custom_params={"l1": 0, "l2": 2}))
        with open(path + "rep4.pkl", "wb") as file:
            pickle.dump(rep4, file)

        dataset = Dataset(filenames=[path + "rep1.pkl", path + "rep2.pkl", path + "rep3.pkl", path + "rep4.pkl"],
                          params={"l1": [0, 1], "l2": [2, 3]})

        return dataset

    def test_execute(self):
        path = EnvironmentSettings.root_path + "test/tmp/semanticmodel/"
        PathBuilder.build(path)
        dataset = self.prepare_dataset(path)

        symbol_table = SymbolTable()
        symbol_table.add("dataset1", SymbolType.DATASET, {"dataset": dataset})
        symbol_table.add("encoding1", SymbolType.ENCODING, {"encoder": Word2VecEncoder,
                                                            "encoder_params": {"k": 3,
                                                                               "model_creator": ModelType.SEQUENCE,
                                                                               "size": 16},
                                                            "dataset": "dataset1"})

        symbol_table.add("report1", SymbolType.REPORT, {"report": SequenceLengthDistribution(), "dataset": "dataset1",
                                                        "params": {"batch_size": 2, "dataset": "dataset1"}})

        model = SemanticModel(path=path)
        model.fill(symbol_table)

        model.execute()
        filename = glob.glob(path + "**/sequence_length_distribution.png")
        self.assertEqual(1, len(filename))
        self.assertTrue("report1" in model._executed)

        shutil.rmtree(path)

        # TODO: make more exhaustive test
