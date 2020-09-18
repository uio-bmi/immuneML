import os
import shutil
from unittest import TestCase

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.subsampling.SubsamplingInstruction import SubsamplingInstruction


class TestSubsamplingInstruction(TestCase):
    def test_run(self):
        path = PathBuilder.build(f"{EnvironmentSettings.tmp_test_path}subsampling/")
        dataset = RandomDatasetGenerator.generate_receptor_dataset(200, labels={"epitope": {"A": 0.5, "B": 0.5}}, path=path,
                                                                   chain_1_length_probabilities={3: 1}, chain_2_length_probabilities={4: 1})
        dataset.name = "d1"

        inst = SubsamplingInstruction(dataset=dataset, subsampled_dataset_sizes=[100, 50], dataset_export_formats=[PickleExporter],
                                      name="subsampling_inst")

        state = inst.run(path + "result/")

        self.assertEqual(2, len(state.subsampled_datasets))
        self.assertEqual(100, state.subsampled_datasets[0].get_example_count())
        self.assertEqual(50, state.subsampled_datasets[1].get_example_count())

        self.assertTrue(all(os.path.isfile(state.subsampled_dataset_paths[name]['pickle'])
                            for name in [dataset.name for dataset in state.subsampled_datasets]))

        shutil.rmtree(path)
