import os
import shutil
from unittest import TestCase

import yaml

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestImmuneMLExporter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_export(self):
        path = EnvironmentSettings.tmp_test_path / "imlexporter/"
        PathBuilder.remove_old_and_build(path)

        repertoires, metadata = RepertoireBuilder.build([["AA"], ["CC"]], path)
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        ImmuneMLExporter.export(dataset, EnvironmentSettings.tmp_test_path / "imlexporter/")

        with open(EnvironmentSettings.tmp_test_path / f"imlexporter/{dataset.name}.yaml", "r") as file:
            dataset2 = yaml.safe_load(file)

        shutil.rmtree(EnvironmentSettings.tmp_test_path / "imlexporter/")

        self.assertTrue(isinstance(dataset2, dict))
        self.assertEqual('RepertoireDataset', dataset2['dataset_class'])
        self.assertEqual(dataset.identifier, dataset2['identifier'])

    def test_export_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "imlexporter_receptor/")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(10, {2: 1}, {3: 1}, {}, path)
        dataset.name = "d1"
        filenames = dataset.get_filenames()
        ImmuneMLExporter.export(dataset, path)

        with open(path / f"{dataset.name}.yaml", "r") as file:
            dataset2 = yaml.safe_load(file)

        self.assertEqual('ReceptorDataset', dataset2['dataset_class'])
        self.assertEqual([str(f) for f in filenames], dataset2['filenames'])

        shutil.rmtree(path)
