import os
import shutil
from unittest import TestCase

import pandas as pd

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.deeprc.DeepRCEncoder import DeepRCEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestDeepRCEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_datasets(self, path: str):
        repertoires, metadata = RepertoireBuilder.build([["A", "B"], ["B", "C"], ["D"], ["E", "F"]], path,
                                                      {"l1": [1, 0, 1, 0], "l2": [2, 3, 2, 3]})

        main_dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        sub_dataset = main_dataset.make_subset([0, 1], path=path, dataset_type="subset")
        return main_dataset, sub_dataset

    def test_encode(self):
        path = EnvironmentSettings.tmp_test_path + "deeprc_encoder/"
        PathBuilder.build(path)
        PathBuilder.build(path+"encoded_data/")

        main_dataset, sub_dataset = self.create_datasets(path)

        enc = DeepRCEncoder.build_object(sub_dataset, **{})

        enc.set_context({"dataset": main_dataset})

        encoded = enc.encode(sub_dataset, EncoderParams(result_path=path+"encoded_data/",
                                                        label_config=LabelConfiguration([Label("l1", [0, 1]), Label("l2", [2, 3])]),
                                                        pool_size=4))

        self.assertListEqual(encoded.encoded_data.example_ids, sub_dataset.get_repertoire_ids())
        self.assertTrue(os.path.isfile(encoded.encoded_data.info["metadata_filepath"]))

        metadata_content = pd.read_csv(encoded.encoded_data.info["metadata_filepath"], sep="\t")
        self.assertListEqual(list(metadata_content["ID"]), sub_dataset.get_repertoire_ids())

        for repertoire in main_dataset.repertoires:
            rep_path = f"{path}/encoded_data/encoding/{repertoire.identifier}.tsv"
            self.assertTrue(os.path.isfile(rep_path))
            repertoire_tsv = pd.read_csv(rep_path, sep="\t")
            self.assertListEqual(list(repertoire_tsv["amino_acid"]), list(repertoire.get_sequence_aas()))

        shutil.rmtree(path)
