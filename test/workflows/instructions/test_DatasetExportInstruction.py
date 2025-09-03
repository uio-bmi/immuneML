import os
import shutil
from glob import glob
from unittest import TestCase

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.CountPerSequenceFilter import CountPerSequenceFilter
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.dataset_generation.DatasetExportInstruction import DatasetExportInstruction
from immuneML.workflows.instructions.dataset_generation.DatasetExportState import DatasetExportState


class TestDatasetExportInstruction(TestCase):
    def test_run(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "dataset_export_instruction/")
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(10, {10: 1}, {12: 1}, {}, path)
        dataset.name = "d1"

        filter = CountPerSequenceFilter(low_count_limit=1, remove_without_count=True, remove_empty_repertoires=True,
                                        batch_size=100)
        instruction = DatasetExportInstruction(datasets=[dataset], preprocessing_sequence=[filter],
                                               name="export_instr", number_of_processes=2)

        result_path = path / "generated/"
        state = instruction.run(result_path=result_path)

        self.assertTrue(isinstance(state, DatasetExportState))
        self.assertEqual(1, len(state.datasets))
        self.assertEqual(1, len(state.formats))
        self.assertEqual("AIRR", state.formats[0])

        self.assertTrue(os.path.isdir(result_path))
        self.assertEqual(2, len(list(glob(str(state.result_path / "*/")))))
        self.assertEqual(1, len(list(glob(str(state.result_path / f"{dataset.name}/*/")))))
        self.assertTrue(os.path.isdir(str(state.result_path / f"{dataset.name}/AIRR/")))
        self.assertTrue(os.path.isfile(str(state.result_path / f"{dataset.name}/AIRR/metadata.csv")))
        self.assertEqual(20, len(list(glob(str(state.result_path / f"{dataset.name}/AIRR/repertoires/*")))))

        shutil.rmtree(path)
