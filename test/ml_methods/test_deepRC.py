import logging
import os
import random as rn
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd
import torch

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.encodings.deeprc.DeepRCEncoder import DeepRCEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestDeepRC(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def get_random_sequence(self, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        return "".join([rn.choice(alphabet) for i in range(rn.choice(range(15, 30)))])

    def make_encoded_data(self, path: Path):
        metadata_filepath = path / f"metadata.tsv"

        rep_ids = [f"REP{i}" for i in range(10)]
        status_label = [chr((i % 2) + 65) for i in range(10)]  # List of alternating strings "A" "B"

        metadata = pd.DataFrame({"ID": rep_ids, "status": status_label})
        metadata.to_csv(sep="\t", index=False, path_or_buf=metadata_filepath)

        for rep_id in rep_ids:
            repertoire_seqs = [self.get_random_sequence() for i in range(100)]

            repertoire_data = pd.DataFrame({"amino_acid": repertoire_seqs,
                                            "templates": [rn.choice(range(1, 1000)) for i in range(100)]})

            repertoire_data.to_csv(sep="\t", index=False, path_or_buf=path / f"{rep_id}.tsv")

        return EncodedData(examples=None, labels={"status": status_label},
                           example_ids=rep_ids, encoding=DeepRCEncoder.__name__,
                           info={"metadata_filepath": metadata_filepath,
                                 "max_sequence_length": 30})

    def dummy_training_function(self, *args, **kwargs):
        """The training function (DeepRC.training_function) only works on GPU.
        To test the rest of the DeepRC class without errors, it is replaced with this dummy function."""
        pass

    def test(self):

        is_installed = True

        try:
            from immuneML.ml_methods.DeepRC import DeepRC
            from deeprc.deeprc_binary.architectures import DeepRC as DeepRCInternal
        except Exception as e:
            is_installed = False

        if is_installed:

            logging.warning("DeepRC test is temporarily excluded")
            path = EnvironmentSettings.tmp_test_path / "deeprc_classifier"
            data_path = path / "encoded_data"
            result_path = path / "result"
            PathBuilder.build(data_path)
            PathBuilder.build(result_path)

            encoded_data = self.make_encoded_data(data_path)
            y = {"status": encoded_data.labels["status"]}

            params = DefaultParamsLoader.load("ml_methods/", "DeepRC")

            classifier = DeepRC(**params)

            # Prepare 'dummy training' for classifier, to test other functionalities
            classifier.result_path = path
            classifier.pytorch_device = torch.device("cpu")
            classifier.training_function = self.dummy_training_function

            train_indices, val_indices = classifier.get_train_val_indices(10)
            self.assertEqual(len(train_indices) + len(val_indices), 10)
            self.assertEqual(set(list(train_indices) + list(val_indices)), set(range(10)))

            # test if 'fit' function saves models
            classifier.fit(encoded_data, "status")

            self.assertListEqual(classifier.get_classes(), ["A", "B"])
            self.assertIsInstance(classifier.model, DeepRCInternal)

            # Test storing and loading of models
            self.assertFalse(classifier.check_if_exists(result_path))
            classifier.store(result_path, feature_names=None)
            self.assertTrue(classifier.check_if_exists(result_path))

            second_classifier = DeepRC(**params)
            second_classifier.load(result_path)

            self.assertIsInstance(second_classifier.model, DeepRCInternal)

            shutil.rmtree(path)

            # test get package info
            params = DefaultParamsLoader.load("ml_methods/", "DeepRC")
            classifier = DeepRC(**params)
            classifier.get_package_info()

        else:
            logging.warning("DeepRC is not installed, skipping test. To install DeepRC, install the requirements from requirements_DeepRC.txt.")
