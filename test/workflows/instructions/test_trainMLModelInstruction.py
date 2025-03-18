import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from immuneML.encodings.word2vec.model_creator.ModelType import ModelType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.ReportConfig import ReportConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.hyperparameter_optimization.strategy.GridSearch import GridSearch
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.ml_methods.classifiers.SVM import SVM
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.preprocessing.filters.ClonesPerRepertoireFilter import ClonesPerRepertoireFilter
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class TestTrainMLModelInstruction(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):

        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "hpoptimproc/")

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

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, labels={"l1": [1, 2], "l2": [0, 1]})
        enc1 = {"k": 3, "model_type": ModelType.SEQUENCE.name, "vector_size": 4, "epochs": 10, 'window': 5}
        enc2 = {"k": 3, "model_type": ModelType.SEQUENCE.name, "vector_size": 6, "epochs": 10, "window": 5}
        hp_settings = [HPSetting(Word2VecEncoder.build_object(dataset, **enc1), enc1,
                                 LogisticRegression(), {"model_selection_cv": False, "model_selection_n_folds": -1},
                                 [], "e1", "ml1"),
                       HPSetting(Word2VecEncoder.build_object(dataset, **enc2), enc2,
                                 SVM(), {"model_selection_cv": False, "model_selection_n_folds": -1},
                                 [ClonesPerRepertoireFilter(lower_limit=-1, upper_limit=1000)], "e2", "ml2", "p2")
                       ]

        report = SequenceLengthDistribution()
        label_config = LabelConfiguration([Label("l1", [1, 2]), Label("l2", [0, 1])])

        process = TrainMLModelInstruction(dataset, GridSearch(hp_settings), hp_settings,
                                          SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig(data_splits={"seqlen": report})),
                                          SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig(data_splits={"seqlen": report})),
                                          {ClassificationMetric.BALANCED_ACCURACY}, ClassificationMetric.BALANCED_ACCURACY, label_config, path,
                                          export_all_ml_settings=True, sequence_type=SequenceType.AMINO_ACID,
                                          region_type=RegionType.IMGT_CDR3)

        state = process.run(result_path=path)

        self.assertTrue(isinstance(state, TrainMLModelState))
        self.assertEqual(1, len(state.assessment_states))
        self.assertTrue("l1" in state.assessment_states[0].label_states)
        self.assertTrue("l2" in state.assessment_states[0].label_states)

        shutil.rmtree(path)
