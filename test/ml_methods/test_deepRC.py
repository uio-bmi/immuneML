import logging
import os
import random as rn
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd
import torch.cuda

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.util.PathBuilder import PathBuilder


class TestDeepRC(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def get_random_sequence(self, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        return "".join([rn.choice(alphabet) for i in range(rn.choice(range(15, 30)))])

    def make_encoded_data(self, path: Path):
        metadata_filepath = path / f"metadata.csv"

        rep_ids = [f"REP{i}" for i in range(10)]
        status_label = [chr((i % 2) + 65) for i in range(10)]  # List of alternating strings "A" "B"

        metadata = pd.DataFrame({"ID": rep_ids, "status": status_label})
        metadata.to_csv(sep=",", index=False, path_or_buf=metadata_filepath)

        for rep_id in rep_ids:
            repertoire_seqs = [self.get_random_sequence() for i in range(100)]

            repertoire_data = pd.DataFrame({"amino_acid": repertoire_seqs,
                                            "templates": [rn.choice(range(1, 1000)) for i in range(100)]})

            repertoire_data.to_csv(sep="\t", index=False, path_or_buf=path / f"{rep_id}.tsv")

        return EncodedData(examples=None, labels={"status": status_label},
                           example_ids=rep_ids, encoding="DeepRCEncoder",
                           info={"metadata_filepath": metadata_filepath,
                                 "max_sequence_length": 30})

    def dummy_training_function(self, *args, **kwargs):
        """The training function (DeepRC.training_function) only works on GPU.
        To test the rest of the DeepRC class without errors, it is replaced with this dummy function."""
        pass

    def internal_deep_RC_test(self):
        from immuneML.ml_methods.classifiers.DeepRC import DeepRC
        from deeprc.architectures import DeepRC as DeepRCInternal

        logging.warning("DeepRC test is temporarily excluded")
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "deeprc_classifier")
        data_path = path / "encoded_data"
        result_path = path / "result"
        PathBuilder.build(data_path)
        PathBuilder.build(result_path)

        encoded_data = self.make_encoded_data(data_path)
        y = {"status": encoded_data.labels["status"]}

        params = DefaultParamsLoader.load("ml_methods/", "DeepRC")
        params['pytorch_device_name'] = 'cuda:1'
        params['n_torch_threads'] = 1
        params['n_workers'] = 1
        params['n_updates'] = 5

        classifier = DeepRC(**params)

        classifier.result_path = path

        train_indices, val_indices = classifier._get_train_val_indices(10, y['status'])
        self.assertEqual(len(train_indices) + len(val_indices), 10)
        self.assertEqual(set(list(train_indices) + list(val_indices)), set(range(10)))

        if torch.cuda.is_available():
            # test if 'fit' function saves models
            classifier.fit(encoded_data, Label("status", values=["A", "B"]))

            self.assertIsInstance(classifier.model, DeepRCInternal)

            # Test storing and loading of models
            classifier.store(result_path, feature_names=None)
            second_classifier = DeepRC(**params)
            second_classifier.load(result_path)

            self.assertIsInstance(second_classifier.model, DeepRCInternal)

        shutil.rmtree(path)

        # test get package info
        params = DefaultParamsLoader.load("ml_methods/", "DeepRC")
        classifier = DeepRC(**params)
        classifier.get_package_info()

    def test(self):
        # the test should always be run
        self.internal_deep_RC_test()
