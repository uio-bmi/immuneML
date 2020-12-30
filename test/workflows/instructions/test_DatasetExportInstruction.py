import os
import shutil
from glob import glob
from unittest import TestCase

from source.IO.dataset_export.AIRRExporter import AIRRExporter
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.dataset_generation.DatasetExportInstruction import DatasetExportInstruction
from source.workflows.instructions.dataset_generation.DatasetExportState import DatasetExportState


class TestDatasetExportInstruction(TestCase):
    def test_run(self):
        path = PathBuilder.build(f"{EnvironmentSettings.tmp_test_path}dataset_export_instruction/")
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(10, {10: 1}, {12: 1}, {}, path)
        dataset.name = "d1"
        instruction = DatasetExportInstruction(datasets=[dataset],
                                                   exporters=[AIRRExporter])

        result_path = f"{path}generated/"
        state = instruction.run(result_path=result_path)

        self.assertTrue(isinstance(state, DatasetExportState))
        self.assertEqual(1, len(state.datasets))
        self.assertEqual(1, len(state.formats))
        self.assertEqual("AIRR", state.formats[0])

        self.assertTrue(os.path.isdir(result_path))
        self.assertEqual(1, len(list(glob(f"{state.result_path}*/"))))
        self.assertEqual(1, len(list(glob(f"{state.result_path}{dataset.name}/*/"))))
        self.assertTrue(os.path.isdir(f"{state.result_path}{dataset.name}/AIRR/"))
        self.assertTrue(os.path.isfile(f"{state.result_path}{dataset.name}/AIRR/metadata.csv"))
        self.assertEqual(10, len(list(glob(f"{state.result_path}{dataset.name}/AIRR/repertoires/*"))))

        shutil.rmtree(path)
