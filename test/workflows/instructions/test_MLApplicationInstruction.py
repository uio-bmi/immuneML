import os
import shutil
from unittest import TestCase

import pandas as pd

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.caching.CacheType import CacheType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFreqRepertoireEncoder import KmerFreqRepertoireEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.ml_methods.SimpleLogisticRegression import SimpleLogisticRegression
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.ml_model_application.MLApplicationInstruction import MLApplicationInstruction


class TestMLApplicationInstruction(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_run(self):

        path = EnvironmentSettings.tmp_test_path + "mlapplicationtest/"
        PathBuilder.build(path)

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(50, {5: 1}, {5: 1}, {"l1": {1: 0.5, 2: 0.5}}, path + 'dataset/')
        ml_method = SimpleLogisticRegression()
        encoder = KmerFreqRepertoireEncoder(NormalizationType.RELATIVE_FREQUENCY, ReadsType.UNIQUE, SequenceEncodingType.CONTINUOUS_KMER, 3,
                                            scale_to_zero_mean=True, scale_to_unit_variance=True)
        label_config = LabelConfiguration([Label("l1", [1, 2])])

        enc_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config, filename="tmp_enc_dataset.pickle", pool_size=4))
        ml_method.fit(enc_dataset.encoded_data, enc_dataset.encoded_data.labels, ['l1'])

        hp_setting = HPSetting(encoder, {"normalization_type": "relative_frequency", "reads": "unique", "sequence_encoding": "continuous_kmer",
                                         "k": 3, "scale_to_zero_mean": True, "scale_to_unit_variance": True}, ml_method, {}, [], 'enc1', 'ml1')

        PathBuilder.build(path+'result/instr1/')
        shutil.copy(path+'dict_vectorizer.pickle', path+'result/instr1/dict_vectorizer.pickle')
        shutil.copy(path+'scaler.pickle', path+'result/instr1/scaler.pickle')

        ml_app = MLApplicationInstruction(dataset, label_config, hp_setting, 4, "instr1", False)
        ml_app.run(path + 'result/')

        predictions_path = path + "result/instr1/predictions.csv"
        self.assertTrue(os.path.isfile(predictions_path))

        df = pd.read_csv(predictions_path)
        self.assertEqual(50, df.shape[0])

        shutil.rmtree(path)
