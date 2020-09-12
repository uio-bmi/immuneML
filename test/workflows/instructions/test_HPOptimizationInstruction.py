import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.strategy.GridSearch import GridSearch
from source.ml_methods.SVM import SVM
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.preprocessing.filters.ClonesPerRepertoireFilter import ClonesPerRepertoireFilter
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class TestHPOptimizationProcess(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):

        path = EnvironmentSettings.tmp_test_path + "hpoptimproc/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build(sequences=[["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"],
                                                                 ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"]], path=path,
                                                      labels={"l1": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                                                                     1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                                              "l2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
                                                                     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, params={"l1": [1, 2], "l2": [0, 1]})
        enc1 = {"k": 3, "model_type": ModelType.SEQUENCE.name, "vector_size": 4}
        enc2 = {"k": 3, "model_type": ModelType.SEQUENCE.name, "vector_size": 6}
        hp_settings = [HPSetting(Word2VecEncoder.build_object(dataset, **enc1), enc1,
                                 SimpleLogisticRegression(), {"model_selection_cv": False, "model_selection_n_folds": -1},
                                 []),
                       HPSetting(Word2VecEncoder.build_object(dataset, **enc2), enc2,
                                 SVM(), {"model_selection_cv": False, "model_selection_n_folds": -1},
                                 [ClonesPerRepertoireFilter(lower_limit=-1, upper_limit=1000)])
                       ]

        report = SequenceLengthDistribution()
        label_config = LabelConfiguration([Label("l1", [1, 2]), Label("l2", [0, 1])])

        process = TrainMLModelInstruction(dataset, GridSearch(hp_settings), hp_settings,
                                          SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig(data_splits={"seqlen": report})),
                                          SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig(data_splits={"seqlen": report})),
                                          {Metric.BALANCED_ACCURACY}, Metric.BALANCED_ACCURACY, label_config, path)

        state = process.run(result_path=path)

        self.assertTrue(isinstance(state, HPOptimizationState))
        self.assertEqual(1, len(state.assessment_states))
        self.assertTrue("l1" in state.assessment_states[0].label_states)
        self.assertTrue("l2" in state.assessment_states[0].label_states)

        shutil.rmtree(path)
