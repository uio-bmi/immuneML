import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.semantic_model.SemanticModel import SemanticModel
from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from immuneML.encodings.word2vec.model_creator.ModelType import ModelType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Metric import Metric
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.ReportConfig import ReportConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.strategy.GridSearch import GridSearch
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class TestSemanticModel(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):

        path = EnvironmentSettings.root_path / "test/tmp/smmodel/"
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
                                    labels={"default": [1, 2]},
                                    metadata_file=metadata)

        label_config = LabelConfiguration()
        label_config.add_label("default", [1, 2])

        hp_settings = [HPSetting(Word2VecEncoder.build_object(dataset, **{"vector_size": 8, "model_type": ModelType.SEQUENCE.name, "k": 3}),
                                 {"vector_size": 8, "model_type": ModelType.SEQUENCE.name, "k": 3},
                                 LogisticRegression(),
                                 {"model_selection_cv": False, "model_selection_n_folds": -1}, [])]

        split_config_assessment = SplitConfig(SplitType.RANDOM, 1, 0.5, ReportConfig())
        split_config_selection = SplitConfig(SplitType.RANDOM, 1, 0.5, ReportConfig())

        instruction = TrainMLModelInstruction(dataset, GridSearch(hp_settings), hp_settings,
                                              split_config_assessment,
                                              split_config_selection,
                                              {Metric.BALANCED_ACCURACY}, Metric.BALANCED_ACCURACY,
                                              label_config, path)
        semantic_model = SemanticModel([instruction], path)

        semantic_model.run()

        shutil.rmtree(path)
