import os
import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.MLProcess import MLProcess


class TestMLProcess(TestCase):
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
        metrics = {MetricType.BALANCED_ACCURACY}
        proc = MLProcess(train_dataset=dataset, test_dataset=dataset, path=path, label_config=label_config,
                         encoder=Word2VecEncoder.create_encoder(dataset, encoder_params), encoder_params=encoder_params,
                         method=SimpleLogisticRegression(), metrics=metrics, optimization_metric=MetricType.ACCURACY,
                         min_example_count=1,
                         ml_params={"model_selection_cv": SplitType.LOOCV, "model_selection_n_folds": 3}, label="l1",
                         ml_score_path=f"{path}score.csv")

        proc.run(1)

        self.assertTrue(os.path.isfile("{}score.csv".format(path)))

        shutil.rmtree(path)
