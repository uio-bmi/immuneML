import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.filtered_sequence_encoding.SequenceCountEncoder import SequenceCountEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestEmersonSequenceCountEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):

        path = EnvironmentSettings.tmp_test_path / "count_encoder/"
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

        label_config = LabelConfiguration([Label("l1", [True, False], positive_class=True)])

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config))

        test = encoded_dataset.encoded_data.examples

        self.assertTrue(test[0] == 1)
        self.assertTrue(test[1] == 1)
        self.assertTrue(test[2] == 0)
        self.assertTrue(test[3] == 0)

        self.assertTrue("III" in encoded_dataset.encoded_data.feature_names)

        shutil.rmtree(path)
