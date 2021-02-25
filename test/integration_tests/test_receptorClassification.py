import os
import pickle
import shutil
from unittest import TestCase

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Metric import Metric
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.config.ReportConfig import ReportConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.strategy.GridSearch import GridSearch
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class TestReceptorClassification(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, path, dataset_size: int = 100):

        receptors = []

        seq1 = ReceptorSequence(amino_acid_sequence="ACACAC")
        seq2 = ReceptorSequence(amino_acid_sequence="DDDEEE")

        for i in range(dataset_size):
            if i % 2 == 0:
                receptors.append(TCABReceptor(alpha=seq1, beta=seq1, metadata={"l1": 1}, identifier=str(i)))
            else:
                receptors.append(TCABReceptor(alpha=seq2, beta=seq2, metadata={"l1": 2}, identifier=str(i)))

        PathBuilder.build(path)
        filename = path / "receptors.pkl"
        with open(filename, "wb") as file:
            pickle.dump(receptors, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = ReceptorDataset(labels={"l1": [1, 2]}, filenames=[filename], identifier="d1")
        return dataset

    def test(self):

        path = EnvironmentSettings.tmp_test_path / "integration_receptor_classification/"
        dataset = self.create_dataset(path)

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
                                              {Metric.BALANCED_ACCURACY}, Metric.BALANCED_ACCURACY, lc, path)

        state = instruction.run(result_path=path)
        print(vars(state))

        self.assertEqual(1.0, state.assessment_states[0].label_states["l1"].optimal_assessment_item.performance[state.optimization_metric.name.lower()])

        shutil.rmtree(path)
