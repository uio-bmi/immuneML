import os
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.hyperparameter_optimization.SplitType import SplitType
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.processes.MLProcess import MLProcess


class TestMLProcess(TestCase):
    # def test_is_ml_possible(self):
    #     path = EnvironmentSettings.root_path + "test/tmp/mlproc_possible/"
    #     lbl_conf = LabelConfiguration()
    #     lbl_conf.add_label("CD", [True, False])
    #     proc = MLProcess(None, "", lbl_conf, None, {}, None, AssessmentType.loocv, [], False, min_example_count=2)
    #     enc_d = Dataset(filenames=RepertoireBuilder.build([[], [], []], path, {"CD": [True, True, False]}),
    #                     params={"CD": {True, False}})
    #
    #     self.assertFalse(proc._is_ml_possible(enc_d))
    #     self.assertWarns(Warning, proc._is_ml_possible, enc_d)
    #
    #     enc_d = Dataset(filenames=RepertoireBuilder.build([[], [], [], []], path, {"CD": [True, True, False, False]}),
    #                     params={"CD": {True, False}})
    #
    #     self.assertTrue(proc._is_ml_possible(enc_d))
    #
    #     shutil.rmtree(path)

    def test_run(self):

        path = EnvironmentSettings.root_path + "test/tmp/mlproc/"

        PathBuilder.build(path)

        filenames, metadata = RepertoireBuilder.build([["AAA"], ["AAA"], ["AAA"], ["AAA"], ["AAA"], ["AAA"]], path,
                                            {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = Dataset(filenames=filenames, params={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        label_config.add_label("l2", [2, 3])
        encoder_params = {
            "k": 3,
            "model_creator": ModelType.SEQUENCE,
            "size": 16
        }
        metrics = {MetricType.BALANCED_ACCURACY}
        proc = MLProcess(train_dataset=dataset, test_dataset=dataset, path=path, label_configuration=label_config,
                         encoder=Word2VecEncoder, encoder_params=encoder_params, method=SimpleLogisticRegression(),
                         metrics=metrics, min_example_count=1, ml_params={"model_selection_cv": SplitType.LOOCV,
                                                                          "model_selection_n_folds": 3})

        proc.run(1)

        self.assertTrue(os.path.isfile("{}ml_details.csv".format(path)))
        # self.assertTrue(os.path.isfile("{}summary.yml".format(path)))
        # with open("{}summary.yml".format(path), "r") as file:
        #     summary = yaml.load(file)
        # self.assertTrue("l1" in summary.keys() and "l2" in summary.keys())
        # self.assertEqual({"min", "max", "mean", "median"}, set(summary["l1"]["balanced_accuracy"]))
        # self.assertEqual({"min", "max", "mean", "median"}, set(summary["l2"]["balanced_accuracy"]))
        # self.assertTrue(all([isinstance(summary["l1"]["balanced_accuracy"][key], float)
        # for key in ["min", "max", "median", "mean"]]))
        # self.assertTrue(all([isinstance(summary["l2"]["balanced_accuracy"][key], float)
        # for key in ["min", "max", "median", "mean"]]))

        dataset = Dataset(filenames=filenames[:3], params={"l1": [0, 1], "l2": [2, 3]})
        proc = MLProcess(train_dataset=dataset, test_dataset=dataset, path=path, label_configuration=label_config,
                         encoder=Word2VecEncoder, encoder_params=encoder_params, method=SimpleLogisticRegression(),
                         metrics=metrics, min_example_count=2, ml_params={"model_selection_cv": SplitType.LOOCV,
                                                                          "model_selection_n_folds": 3})

        with self.assertWarns(Warning):
            proc.run(2)

        shutil.rmtree(path)
