import os
import shutil
from unittest import TestCase

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.util.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.ReportConfig import ReportConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.strategy.GridSearch import GridSearch
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class TestSequenceClassification(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test(self):

        path = EnvironmentSettings.tmp_test_path / "integration_sequence_classification/"
        dataset = RandomDatasetGenerator.generate_sequence_dataset(50, {4: 1}, {'l1': {1: 0.5, 2: 0.5}}, path / 'data')

        os.environ["cache_type"] = "test"
        encoder_params = {
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "sequence_type": SequenceType.AMINO_ACID.name,
            "k": 3
        }

        hp_setting = HPSetting(encoder=KmerFrequencyEncoder.build_object(dataset, **encoder_params), encoder_params=encoder_params,
                               ml_method=LogisticRegression(), ml_params={"model_selection_cv": False, "model_selection_n_folds": -1},
                               preproc_sequence=[])

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        instruction = TrainMLModelInstruction(dataset, GridSearch([hp_setting]), [hp_setting],
                                              SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig()),
                                              SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig()),
                                              {ClassificationMetric.BALANCED_ACCURACY}, ClassificationMetric.BALANCED_ACCURACY, lc, path)

        result = instruction.run(result_path=path)

        shutil.rmtree(path)
