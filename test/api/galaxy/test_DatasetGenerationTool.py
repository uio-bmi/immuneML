import os
import shutil
from argparse import Namespace
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import run_immuneML
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestDatasetGenerationTool(TestCase):

    def prepare_specs(self, path):
        specs = {'definitions': {
                    "datasets": {
                        "d1": {
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
            },
            "instructions": {
                "inst1": {"type": "DatasetExport", "export_formats": ["ImmuneML"], "datasets": ["d1"]}
            }
        }

        with open(path, "w") as file:
            yaml.dump(specs, file)

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path / "galaxy_api_dataset_generation/"
        PathBuilder.build(path)
        yaml_path = path / "specs.yaml"
        result_path = path / "results/"

        PathBuilder.build(path)
        self.prepare_specs(yaml_path)

        run_immuneML(Namespace(**{"specification_path": yaml_path, "result_path": result_path, 'tool': "DatasetGenerationTool"}))

        self.assertTrue(os.path.isfile(result_path / "result/dataset_metadata.csv"))
        self.assertTrue(os.path.isfile(result_path / "result/dataset.iml_dataset"))
        self.assertEqual(200, len([name for name in os.listdir(result_path / "result/repertoires/")
                                   if os.path.isfile(os.path.join(result_path / "result/repertoires/", name))]))

        shutil.rmtree(path)
