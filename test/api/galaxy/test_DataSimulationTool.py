import os
import shutil
from argparse import Namespace
from unittest import TestCase

import yaml

from source.app.ImmuneMLApp import run_immuneML
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestDataSimulationTool(TestCase):
    def test_run(self):

        path = EnvironmentSettings.tmp_test_path / "galaxy_api_dataset_simulation/"
        PathBuilder.build(path)

        yaml_path = path / "specs.yaml"
        result_path = path / "results/"

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
                "inst1": {"type": "DatasetExport", "export_formats": ["Pickle"], "datasets": ["d1"]}
            }
        }

        with open(yaml_path, "w") as file:
            yaml.dump(specs, file)

        PathBuilder.build(path)

        run_immuneML(Namespace(**{"specification_path": yaml_path, "result_path": result_path, 'tool': "DataSimulationTool"}))

        self.assertTrue(os.path.isfile(result_path / "result/d1_metadata.csv"))
        self.assertTrue(os.path.isfile(result_path / "result/d1.iml_dataset"))
        self.assertEqual(200, len([name for name in os.listdir(result_path / "result/repertoires/")
                                   if os.path.isfile(os.path.join(result_path / "result/repertoires/", name))]))

        shutil.rmtree(path)
