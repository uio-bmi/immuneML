import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.MLProcess import MLProcess


class TestMLProcess(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):
        path = EnvironmentSettings.root_path + "test/tmp/mlproc/"

        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAA"], ["AAA"], ["AAA"], ["AAA"], ["AAA"]], path,
                                                        {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, params={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        encoder_params = {
            "k": 3,
            "model_type": ModelType.SEQUENCE.name,
            "vector_size": 16
        }
        metrics = {Metric.BALANCED_ACCURACY}
        proc = MLProcess(train_dataset=dataset, test_dataset=dataset, path=path, label_config=label_config,
                         hp_setting=HPSetting(encoder=Word2VecEncoder.build_object(dataset, **encoder_params), encoder_params=encoder_params,
                                              ml_method=SimpleLogisticRegression(),
                                              ml_params={"model_selection_cv": SplitType.LOOCV, "model_selection_n_folds": 3}, preproc_sequence=[]),
                         metrics=metrics, optimization_metric=Metric.ACCURACY, label="l1")

        proc.run(1)

        self.assertTrue(os.path.isfile("{}ml_score.csv".format(path)))

        shutil.rmtree(path)
