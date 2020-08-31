import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.filtered_sequence_encoding.SequenceCountEncoder import SequenceCountEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestEmersonSequenceCountEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):

        path = EnvironmentSettings.tmp_test_path + "count_encoder/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                       ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                       ["CCC", "FFF", "MMM"],
                                                       ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
                                                      labels={"l1": [True, True, False, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, identifier="1")

        encoder = SequenceCountEncoder.build_object(dataset, **{
            "comparison_attributes": ["sequence_aas"],
            "p_value_threshold": 0.4, "sequence_batch_size": 4
        })

        label_config = LabelConfiguration([Label("l1", [True, False])])

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config))

        test = encoded_dataset.encoded_data.examples

        self.assertTrue(test[0, 0] == 1)
        self.assertTrue(test[1, 0] == 1)
        self.assertTrue(test[0, 1] == 0)
        self.assertTrue(test[1, 1] == 0)
        self.assertTrue(test[3, 1] == 1)

        self.assertTrue("III" in encoded_dataset.encoded_data.feature_names)
        self.assertTrue("CCC" in encoded_dataset.encoded_data.feature_names)

        shutil.rmtree(path)
