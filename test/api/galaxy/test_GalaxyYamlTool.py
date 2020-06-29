import os
import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.api.galaxy.GalaxyYamlTool import GalaxyYamlTool
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder


class TestGalaxyYamlTool(TestCase):
    def test_run(self):

        path = PathBuilder.build(f"{EnvironmentSettings.tmp_test_path}api_galaxy_yaml_tool/")
        result_path = f"{path}result/"

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(10, {10: 1}, {12: 1}, {}, result_path)
        dataset.name = "d1"
        PickleExporter.export(dataset, result_path)

        specs = {
            "definitions": {
                "datasets": {
                    "new_d1": {
                        "format": "Pickle",
                        "params": {
                            "metadata_file": f"{result_path}d1_metadata.csv"
                        }
                    },
                    "d2": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 10,
                            "sequence_length_probabilities": {10: 1},
                            'sequence_count_probabilities': {10: 1},
                            'labels': {}
                        }
                    }
                }
            },
            "instructions": {
                "inst1": {
                    "type": "DatasetGeneration",
                    "datasets": ["new_d1", 'd2'],
                    "formats": ["AIRR"]
                }
            }
        }

        specs_path = f"{path}specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        tool = GalaxyYamlTool(specs_path, result_path + "result/")
        tool.start_path = path
        tool.run()

        self.assertTrue(os.path.exists(f"{result_path}result/inst1/new_d1/AIRR"))
        self.assertTrue(os.path.exists(f"{result_path}result/inst1/d2/AIRR"))
        self.assertTrue(os.path.exists(f"{result_path}result/d2"))

        shutil.rmtree(path)
