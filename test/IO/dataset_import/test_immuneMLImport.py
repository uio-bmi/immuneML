import shutil
from pathlib import Path
from unittest import TestCase

import yaml

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.IO.dataset_import.ImmuneMLImport import ImmuneMLImport
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestImmuneMLImport(TestCase):

    def test_import_repertoires(self):
        base_path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "iml_import_repertoires/")
        orig_dataset_folder = base_path / "orig_dataset"
        orig_dataset = RandomDatasetGenerator.generate_repertoire_dataset(2, {10: 1}, {3: 1}, {}, orig_dataset_folder)
        dataset_path = ImmuneMLExporter.export(orig_dataset, orig_dataset_folder)

        imported_dataset = ImmuneMLImport.import_dataset({"path": dataset_path}, "dataset_name")

        self.assertEqual(2, len(imported_dataset.get_data()))
        self.assertListEqual(orig_dataset.get_example_ids(), imported_dataset.get_example_ids())

        # testing if dataset can be imported from different folder location
        moved_dataset_folder = base_path / "newpath" / "subfolder"
        shutil.move(orig_dataset_folder, moved_dataset_folder)

        moved_imported_dataset = ImmuneMLImport.import_dataset({"path": moved_dataset_folder / dataset_path.name},
                                                               "dataset_name")
        self.assertEqual(2, len(moved_imported_dataset.get_data()))
        self.assertListEqual(orig_dataset.get_example_ids(), moved_imported_dataset.get_example_ids())

        shutil.rmtree(base_path)

    def test_import_receptors(self):
        base_path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "iml_import_receptors/")
        orig_dataset_folder = base_path / "orig_dataset"

        orig_dataset = RandomDatasetGenerator.generate_receptor_dataset(10, {2: 1}, {3: 1}, {}, orig_dataset_folder)
        dataset_path = ImmuneMLExporter.export(orig_dataset, orig_dataset_folder)

        imported_dataset = ImmuneMLImport.import_dataset({"path": dataset_path}, "d2")

        self.assertEqual(orig_dataset.get_example_count(), imported_dataset.get_example_count())
        self.assertEqual(len(list(imported_dataset.get_data())), len(list(imported_dataset.get_data())))
        self.assertListEqual(orig_dataset.get_example_ids(), imported_dataset.get_example_ids())

        # testing if dataset can be imported from different folder location
        moved_dataset_folder = base_path / "newpath" / "subfolder"
        shutil.move(orig_dataset_folder, moved_dataset_folder)

        moved_imported_dataset = ImmuneMLImport.import_dataset({"path": moved_dataset_folder / dataset_path.name},
                                                               "dataset_name")
        self.assertEqual(orig_dataset.get_example_count(), moved_imported_dataset.get_example_count())
        self.assertListEqual(orig_dataset.get_example_ids(), moved_imported_dataset.get_example_ids())

        # original data has been deleted so get_data doesnt work, testing this way instead
        self.assertEqual(orig_dataset.get_example_count(), len(list(moved_imported_dataset.get_data())))

        shutil.rmtree(base_path)
