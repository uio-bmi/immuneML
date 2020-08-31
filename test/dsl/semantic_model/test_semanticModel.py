import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.semantic_model.SemanticModel import SemanticModel
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.hyperparameter_optimization.strategy.GridSearch import GridSearch
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.HPOptimizationInstruction import HPOptimizationInstruction


class TestSemanticModel(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

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

        hp_settings = [HPSetting(Word2VecEncoder.build_object(dataset, **{"vector_size": 8, "model_type": ModelType.SEQUENCE.name, "k": 3}),
                                 {"vector_size": 8, "model_type": ModelType.SEQUENCE.name, "k": 3},
                                 SimpleLogisticRegression(),
                                 {"model_selection_cv": False, "model_selection_n_folds": -1}, [])]

        split_config_assessment = SplitConfig(SplitType.RANDOM, 1, 0.5, ReportConfig())
        split_config_selection = SplitConfig(SplitType.RANDOM, 1, 0.5, ReportConfig())

        instruction = HPOptimizationInstruction(dataset, GridSearch(hp_settings), hp_settings,
                                                split_config_assessment,
                                                split_config_selection,
                                                {Metric.BALANCED_ACCURACY}, Metric.BALANCED_ACCURACY,
                                                label_config, path)
        semantic_model = SemanticModel([instruction], path)

        semantic_model.run()

        shutil.rmtree(path)
