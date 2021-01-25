import os
import pickle
import shutil
from unittest import TestCase

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
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
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


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
        filename = path / "sequences.pkl"
        with open(filename, "wb") as file:
            pickle.dump(sequences, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = SequenceDataset(params={"l1": [1, 2]}, filenames=[filename], identifier="d1")
        return dataset

    def test(self):

        path = EnvironmentSettings.tmp_test_path / "integration_sequence_classification/"
        dataset = self.create_dataset(path)

        os.environ["cache_type"] = "test"
        encoder_params = {
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
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
                                              {Metric.BALANCED_ACCURACY}, Metric.BALANCED_ACCURACY, lc, path)

        result = instruction.run(result_path=path)

        shutil.rmtree(path)


