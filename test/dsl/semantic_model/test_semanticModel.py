import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.semantic_model.SemanticModel import SemanticModel
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.hyperparameter_optimization.strategy.GridSearch import GridSearch
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.processes.HPOptimizationProcess import HPOptimizationProcess


class TestSemanticModel(TestCase):

    def test_run(self):

        path = EnvironmentSettings.root_path + "test/tmp/smmodel/"
        PathBuilder.build(path)
        repertoires, metadata = RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"],
                                                       ["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"],
                                                       ["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"],
                                                       ["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"],
                                                       ["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"],
                                                       ["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"],
                                                       ["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"],
                                                       ["AAA", "CCC"], ["TTTT"], ["AAA", "CCC"], ["TTTT"]], path,
                                                      {"default": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                                                                   1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]})
        dataset = RepertoireDataset(repertoires=repertoires,
                                    params={"default": [1, 2]},
                                    metadata_file=metadata)

        label_config = LabelConfiguration()
        label_config.add_label("default", [1, 2])

        hp_settings = [HPSetting(Word2VecEncoder, {"vector_size": 8, "model_type": ModelType.SEQUENCE, "k": 3},
                                 SimpleLogisticRegression(),
                                 {"model_selection_cv": False, "model_selection_n_folds": -1}, [])]

        instruction = HPOptimizationProcess(dataset, GridSearch(hp_settings), hp_settings,
                                            SplitConfig(SplitType.RANDOM, 1, 0.5, "default", ReportConfig()),
                                            SplitConfig(SplitType.RANDOM, 1, 0.5, "default", ReportConfig()),
                                            {MetricType.BALANCED_ACCURACY},
                                            label_config, path)
        semantic_model = SemanticModel([instruction], path)

        semantic_model.run()

        shutil.rmtree(path)
