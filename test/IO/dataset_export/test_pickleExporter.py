import os
import pickle
import shutil
from unittest import TestCase

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheType import CacheType
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
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
        PickleExporter.export(dataset, EnvironmentSettings.tmp_test_path + "pickleexporter/")

        with open(EnvironmentSettings.tmp_test_path + f"pickleexporter/{dataset.name}.iml_dataset", "rb") as file:
            dataset2 = pickle.load(file)

        shutil.rmtree(EnvironmentSettings.tmp_test_path + "pickleexporter/")

        self.assertTrue(isinstance(dataset2, RepertoireDataset))
        self.assertEqual(2, len(dataset2.get_data()))
        self.assertEqual("rep_0", dataset2.get_data()[0].metadata["donor"])

    def test_export_receptor_dataset(self):
        path = EnvironmentSettings.tmp_test_path + "pickleexporter_receptor/"
        PathBuilder.build(path)

        dataset = RandomDatasetGenerator.generate_receptor_dataset(10, {2: 1}, {3: 1}, {}, path)
        dataset.name = "d1"
        PickleExporter.export(dataset, path)

        with open(f"{path}/{dataset.name}.iml_dataset", "rb") as file:
            dataset2 = pickle.load(file)

        self.assertTrue(isinstance(dataset2, ReceptorDataset))
        self.assertEqual(10, dataset2.get_example_count())

        shutil.rmtree(path)
