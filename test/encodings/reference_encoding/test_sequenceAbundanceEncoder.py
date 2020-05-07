import os
import shutil
from unittest import TestCase

import numpy as np

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestSequenceAbundanceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):

        path = EnvironmentSettings.tmp_test_path + "abundanceencoder/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                       ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                       ["CCC", "FFF", "MMM"],
                                                       ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
                                                      labels={"l1": [True, True, False, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, identifier="1")

        encoder = SequenceAbundanceEncoder.build_object(dataset, **{
            "comparison_attributes": ["sequence_aas"],
            "p_value_threshold": 0.4, "pool_size": 4
        })

        label_config = LabelConfiguration([Label("l1", [True, False])])

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_configuration=label_config,
                                                                filename="encoded.pickle"))

        self.assertTrue(np.array_equal(np.array([[1, 4], [1, 6], [1, 3], [1, 6]]), encoded_dataset.encoded_data.examples))

        encoder.p_value_threshold = 0.05

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_configuration=label_config,
                                                                filename="encoded.pickle"))

        self.assertTrue(np.array_equal(np.array([[0, 4], [0, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

        shutil.rmtree(path)
