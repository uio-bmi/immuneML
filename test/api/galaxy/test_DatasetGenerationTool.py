import os
import shutil
from unittest import TestCase

import yaml

from source.api.galaxy.DatasetGenerationTool import DatasetGenerationTool
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestDatasetGenerationTool(TestCase):

    def prepare_specs(self, path):
        specs = {
            "dataset_name": {
                "format": "RandomRepertoireDataset",
                "params": {
                    "repertoire_count": 100,
                    "sequence_count_probabilities": {
                        100: 1
                    },
                    "sequence_length_probabilities": {
                        10: 1
                    },
                    "labels": {}
                }
            }
        }

        with open(path, "w") as file:
            yaml.dump(specs, file)

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path + "galaxy_api_dataset_generation/"
        PathBuilder.build(path)
        yaml_path = f"{path}specs.yaml"
        result_path = f"{path}results/"

        PathBuilder.build(path)
        self.prepare_specs(yaml_path)

        tool = DatasetGenerationTool(yaml_path=yaml_path, output_dir=result_path)
        tool.run()

        self.assertTrue(os.path.isfile(f"{result_path}metadata.csv"))
        self.assertEqual(202, len([name for name in os.listdir(result_path) if os.path.isfile(os.path.join(result_path, name))]))

        shutil.rmtree(path)
