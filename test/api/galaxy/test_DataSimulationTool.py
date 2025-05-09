import os
import shutil
from argparse import Namespace
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import run_immuneML
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestDataSimulationTool(TestCase):
    def test_run(self):

        path = EnvironmentSettings.tmp_test_path / "galaxy_api_dataset_simulation/"
        PathBuilder.remove_old_and_build(path)

        yaml_path = path / "specs.yaml"
        result_path = path / "results/"

        specs = {'definitions': {
            "datasets": {
                "dataset": {
                    "format": "RandomRepertoireDataset",
                    "params": {
                        "repertoire_count": 10,
                        "sequence_count_probabilities": {
                            10: 1
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
                "inst1": {"type": "DatasetExport", "export_formats": ["AIRR"], "datasets": ["dataset"]}
            }
        }

        with open(yaml_path, "w") as file:
            yaml.dump(specs, file)

        run_immuneML(Namespace(**{"specification_path": yaml_path, "result_path": result_path, 'tool': "DataSimulationTool"}))

        self.assertTrue(os.path.isfile(result_path / "galaxy_dataset/dataset_metadata.csv"))
        self.assertTrue(os.path.isfile(result_path / "galaxy_dataset/dataset.yaml"))
        self.assertEqual(20, len([name for name in os.listdir(result_path / "galaxy_dataset/repertoires/")
                                   if os.path.isfile(os.path.join(result_path / "galaxy_dataset/repertoires/", name))]))

        shutil.rmtree(path)

    def test_run_sequence(self):

        path = EnvironmentSettings.tmp_test_path / "galaxy_api_dataset_simulation_seq/"
        PathBuilder.remove_old_and_build(path)

        yaml_path = path / "specs.yaml"
        result_path = path / "results/"

        specs = {'definitions': {
            "datasets": {
                "dataset": {
                    "format": "RandomSequenceDataset",
                    "params": {
                        "sequence_count": 10
                    }
                }
            }
        },
            "instructions": {
                "inst1": {"type": "DatasetExport", "export_formats": ["AIRR"], "datasets": ["dataset"]}
            }
        }

        with open(yaml_path, "w") as file:
            yaml.dump(specs, file)

        run_immuneML(Namespace(**{"specification_path": yaml_path, "result_path": result_path, 'tool': "DataSimulationTool"}))

        self.assertTrue(os.path.isfile(result_path / "galaxy_dataset/dataset.tsv"))
        self.assertTrue(os.path.isfile(result_path / "galaxy_dataset/dataset.yaml"))

        shutil.rmtree(path)

