import os
import pickle
import shutil
from unittest import TestCase

import yaml

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.dsl.AssessmentType import AssessmentType
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.util.PathBuilder import PathBuilder
from source.workflows.processes.MLProcess import MLProcess


class TestMLProcess(TestCase):
    def test_run(self):

        path = EnvironmentSettings.root_path + "test/tmp/mlproc/"

        PathBuilder.build(path)

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
        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        label_config.add_label("l2", [2, 3])
        encoder_params = {
            "k": 3,
            "model_creator": ModelType.SEQUENCE,
            "size": 16
        }
        metrics = [MetricType.BALANCED_ACCURACY]
        proc = MLProcess(dataset=dataset, split_count=2, path=path, label_configuration=label_config,
                         encoder=Word2VecEncoder, encoder_params=encoder_params, method=SimpleLogisticRegression(),
                         assessment_type=AssessmentType.loocv, metrics=metrics, model_selection_cv=False)

        proc.run()

        self.assertTrue(os.path.isfile("{}loocv/ml_details.csv".format(path)))
        self.assertTrue(os.path.isfile("{}loocv/summary.yml".format(path)))
        with open("{}loocv/summary.yml".format(path), "r") as file:
            summary = yaml.load(file)
        self.assertTrue("l1" in summary.keys() and "l2" in summary.keys())
        self.assertEqual({"min", "max", "mean", "median"}, set(summary["l1"]["balanced_accuracy"]))
        self.assertEqual({"min", "max", "mean", "median"}, set(summary["l2"]["balanced_accuracy"]))
        self.assertTrue(all([isinstance(summary["l1"]["balanced_accuracy"][key], float) for key in ["min", "max", "median", "mean"]]))
        self.assertTrue(all([isinstance(summary["l2"]["balanced_accuracy"][key], float) for key in ["min", "max", "median", "mean"]]))

        shutil.rmtree(path)
