import os
import shutil
from unittest import TestCase

import pandas as pd

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
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.config.SplitType import SplitType
from source.hyperparameter_optimization.strategy.GridSearch import GridSearch
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.reports.ReportResult import ReportResult
from source.reports.hyperparameter.MLSettingsPerformance import MLSettingsPerformance
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class TestMLSettingsPerformance(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_state_object(self, path):
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
                                                                   ["AAA", "CCC", "DDD"], ["AAA", "CCC", "DDD"]],
                                                        path=path,
                                                        labels={
                                                            "l1": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                                                                   1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                                            "l2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
                                                                   0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata,
                                    params={"l1": [1, 2], "l2": [0, 1]})
        enc_params = {"k": 3, "model_type": ModelType.SEQUENCE.name, "vector_size": 4}
        hp_settings = [HPSetting(Word2VecEncoder.build_object(dataset, **enc_params), enc_params,
                                 SimpleLogisticRegression(),
                                 {"model_selection_cv": False, "model_selection_n_folds": -1},
                                 [])]

        label_config = LabelConfiguration([Label("l1", [1, 2]), Label("l2", [0, 1])])

        process = TrainMLModelInstruction(dataset, GridSearch(hp_settings), hp_settings,
                                          SplitConfig(SplitType.RANDOM, 1, 0.5),
                                          SplitConfig(SplitType.RANDOM, 1, 0.5),
                                          {Metric.BALANCED_ACCURACY}, Metric.BALANCED_ACCURACY, label_config, path)

        state = process.run(result_path=path)

        return state

    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/mlsettingsperformance/"
        PathBuilder.build(path)

        report = MLSettingsPerformance()

        report.result_path = path
        report.state = self._create_state_object(path + "input_data/")

        report.check_prerequisites()
        result = report.generate()

        self.assertTrue(os.path.isfile(path + "performance.csv"))
        self.assertTrue(os.path.isfile(path + "performance.html"))

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_figures[0].path, path + "performance.html")
        self.assertEqual(result.output_tables[0].path, path + "performance.csv")

        written_data = pd.read_csv(path + "performance.csv")
        self.assertEqual(list(written_data.columns), ["fold", "label", "encoding", "ml_method", "performance"])

        shutil.rmtree(path)

    def test_plot(self):
        # Does not assert anything, but can be used to manually check if the plot looks like it should

        path = EnvironmentSettings.root_path + "test/tmp/mlsettingsperformance/"
        PathBuilder.build(path)

        report = MLSettingsPerformance()

        report.result_path = path
        report.state = self._create_state_object(path + "input_data/")

        df = pd.DataFrame({"fold": [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                      "label": ["l1", "l1", "l1", "l1", "l2", "l2", "l2", "l2", "l1", "l1", "l1", "l1", "l2", "l2", "l2", "l2"],
                      report.vertical_grouping: ["e1", "e2", "e1", "e2", "e1", "e2", "e1", "e2", "e1", "e2", "e1", "e2", "e1", "e2", "e1", "e2"],
                      "ml_method": ["ml1", "ml1", "ml2", "ml2", "ml1", "ml1", "ml2", "ml2", "ml1", "ml1", "ml2", "ml2", "ml1", "ml1", "ml2", "ml2"],
                      "performance": [0.5, 0.8, 0.4, 0.8, 0.9, 0.2, 0.5, 0.6, 0.8, 0.4, 0.8, 0.9, 0.2, 0.5, 0.6, 0.5]})

        report._plot(df)

        shutil.rmtree(path)

