import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.strategy.GridSearch import GridSearch
from source.ml_methods.SVM import SVM
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.preprocessing.filters.ClonotypeCountFilter import ClonotypeCountFilter
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.HPOptimizationInstruction import HPOptimizationInstruction


class TestHPOptimizationProcess(TestCase):

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
        hp_settings = [HPSetting(Word2VecEncoder, {"k": 3, "model_type": ModelType.SEQUENCE, "vector_size": 4},
                                 SimpleLogisticRegression(), {"model_selection_cv": False, "model_selection_n_folds": -1},
                                 []),
                       HPSetting(Word2VecEncoder, {"k": 3, "model_type": ModelType.SEQUENCE, "vector_size": 6},
                                 SVM(), {"model_selection_cv": False, "model_selection_n_folds": -1},
                                 [ClonotypeCountFilter(lower_limit=-1, upper_limit=1000)])
                       ]

        report = SequenceLengthDistribution()
        label_config = LabelConfiguration([Label("l1", [1, 2]), Label("l2", [0, 1])])

        process = HPOptimizationInstruction(dataset, GridSearch(hp_settings), hp_settings,
                                            SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig(data_splits=[report])),
                                            SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig(data_splits=[report])),
                                            {MetricType.BALANCED_ACCURACY}, label_config, path)

        state = process.run(result_path=path)

        self.assertTrue(isinstance(state, HPOptimizationState))
        self.assertEqual(1, len(state.assessment_states))
        self.assertTrue("l1" in state.assessment_states[0].label_states)
        self.assertTrue("l2" in state.assessment_states[0].label_states)

        shutil.rmtree(path)
