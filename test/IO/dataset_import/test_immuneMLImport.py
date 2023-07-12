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
    def test_import(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "iml_import/")

        repertoires, metadata = RepertoireBuilder.build([["AA"], ["CC"]], path)
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)

        with open(path / "dataset.yaml", "w") as file:
            dataset_dict = {key: item if not isinstance(item, Path) else str(item) for key, item in vars(dataset).items()
                            if key not in ['repertoires', 'encoded_data']}
            yaml.dump({**dataset_dict, **{"dataset_class": "RepertoireDataset"}}, file)

        dataset2 = ImmuneMLImport.import_dataset({"path": path / "dataset.yaml"}, "dataset_name")

        shutil.rmtree(path)

        self.assertEqual(2, len(dataset2.get_data()))
        self.assertEqual("rep_1", dataset2.get_data()[1].metadata["subject_id"])

    def test_import_receptors(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "iml_import_receptors/")

        dataset = RandomDatasetGenerator.generate_receptor_dataset(10, {2: 1}, {3: 1}, {}, path)
        dataset.name = "d1"
        ImmuneMLExporter.export(dataset, path)

        receptor_dataset = ImmuneMLImport.import_dataset({"path": path / "d1.yaml"}, "d2")

        self.assertEqual(10, len(list(receptor_dataset.get_data())))

        shutil.rmtree(path)
