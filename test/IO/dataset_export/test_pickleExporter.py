import os
import pickle
import shutil
from unittest import TestCase

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestPickleExporter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_export(self):
        path = EnvironmentSettings.tmp_test_path + "pickleexporter/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["AA"], ["CC"]], path)
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        PickleExporter.export(dataset, EnvironmentSettings.tmp_test_path + "pickleexporter/", "dataset.pkl")

        with open(EnvironmentSettings.tmp_test_path + "pickleexporter/dataset.pkl", "rb") as file:
            dataset2 = pickle.load(file)

        shutil.rmtree(EnvironmentSettings.tmp_test_path + "pickleexporter/")

        self.assertTrue(isinstance(dataset2, RepertoireDataset))
        self.assertEqual(2, len(dataset2.get_data()))
        self.assertEqual("rep_0", dataset2.get_data()[0].metadata["donor"])
