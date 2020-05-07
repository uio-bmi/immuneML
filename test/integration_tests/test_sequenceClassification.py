import os
import pickle
import shutil
from unittest import TestCase

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.caching.CacheType import CacheType
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.Constants import Constants
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
from source.workflows.instructions.HPOptimizationInstruction import HPOptimizationInstruction


class TestSequenceClassification(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, path, dataset_size: int = 50):

        sequences = []

        for i in range(dataset_size):
            if i % 2 == 0:
                sequences.append(ReceptorSequence(amino_acid_sequence="AAACCC",
                                                  identifier=str(i),
                                                  metadata=SequenceMetadata(custom_params={"l1": 1})))
            else:
                sequences.append(ReceptorSequence(amino_acid_sequence="ACACAC",
                                                  identifier=str(i),
                                                  metadata=SequenceMetadata(custom_params={"l1": 2})))

        PathBuilder.build(path)
        filename = "{}sequences.pkl".format(path)
        with open(filename, "wb") as file:
            pickle.dump(sequences, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = SequenceDataset(params={"l1": [1, 2]}, filenames=[filename], identifier="d1")
        return dataset

    def test(self):

        path = EnvironmentSettings.tmp_test_path + "integration_sequence_classification/"
        dataset = self.create_dataset(path)

        os.environ["cache_type"] = "test"

        hp_setting = HPSetting(encoder=KmerFrequencyEncoder, encoder_params={
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "k": 3
        }, ml_method=SimpleLogisticRegression(), ml_params={"model_selection_cv": False, "model_selection_n_folds": -1},
                               preproc_sequence=[])

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        instruction = HPOptimizationInstruction(dataset, GridSearch([hp_setting]), [hp_setting],
                                                SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig()),
                                                SplitConfig(SplitType.RANDOM, 1, 0.5, reports=ReportConfig()),
                                                {MetricType.BALANCED_ACCURACY}, MetricType.BALANCED_ACCURACY, lc, path)

        result = instruction.run(result_path=path)
        print(result)

        shutil.rmtree(path)


