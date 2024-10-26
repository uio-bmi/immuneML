import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFreqRepertoireEncoder import KmerFreqRepertoireEncoder
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.util.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.ml_model_application.MLApplicationInstruction import MLApplicationInstruction


class TestMLApplicationInstruction(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):

        path = EnvironmentSettings.tmp_test_path / "mlapplicationtest/"
        PathBuilder.build(path)

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(50, {5: 1}, {5: 1}, {"l1": {1: 0.5, 2: 0.5}}, path / 'training_dataset/')
        test_dataset = RandomDatasetGenerator.generate_repertoire_dataset(20, {5: 1}, {5: 1}, {"l1": {1: 0.5, 2: 0.5}}, path / 'test_dataset/')
        ml_method = LogisticRegression()
        encoder = KmerFreqRepertoireEncoder(NormalizationType.RELATIVE_FREQUENCY, ReadsType.UNIQUE, SequenceEncodingType.CONTINUOUS_KMER, 3,
                                            scale_to_zero_mean=True, scale_to_unit_variance=True)
        label = Label("l1", [1, 2])
        label_config = LabelConfiguration([label])

        enc_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config, pool_size=4,
                                                            region_type=RegionType.IMGT_CDR3, sequence_type=SequenceType.AMINO_ACID))
        ml_method.fit(enc_dataset.encoded_data, label)

        hp_setting = HPSetting(encoder, {"normalization_type": "relative_frequency", "reads": "unique", "sequence_encoding": "continuous_kmer",
                                         "k": 3, "scale_to_zero_mean": True, "scale_to_unit_variance": True}, ml_method, {}, [], 'enc1', 'ml1')

        PathBuilder.build(path / 'result/instr1/')

        ml_app = MLApplicationInstruction(test_dataset, label_config, hp_setting,
                                          [ClassificationMetric.get_metric("accuracy"), ClassificationMetric.get_metric("precision"),
                                           ClassificationMetric.get_metric("recall"), ClassificationMetric.get_metric("auc")], 4, "instr1")
        ml_app.run(path / 'result/')

        predictions_path = path / "result/instr1/predictions.csv"
        self.assertTrue(os.path.isfile(predictions_path))
        metrics_path = path / "result/instr1/metrics.csv"
        self.assertTrue(os.path.isfile(predictions_path))

        df = pd.read_csv(predictions_path)
        self.assertEqual(20, df.shape[0])

        df = pd.read_csv(metrics_path)
        self.assertEqual(sorted(df.columns), sorted(["accuracy", "precision", "recall", "auc"]))
        self.assertEqual(1, df.shape[0])

        shutil.rmtree(path)
