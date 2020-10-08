import shutil
from unittest import TestCase

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestDatasetGenerationHTMLOutput(TestCase):

    def test(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "integration_dataset_gen_html/")
        dataset_path = f"{path}initial_dataset/"

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 10,
                            "sequence_count_probabilities": {
                                10: 1
                            },
                            "sequence_length_probabilities": {
                                12: 1
                            },
                            "labels": {},
                            "result_path": dataset_path
                        }
                    }
                }
            },
            "instructions": {"instr1": {"type": "DatasetGeneration", "export_formats": ["Pickle", "AIRR"], "datasets": ["d1"]}},
            "output": {"format": "HTML"}
        }

        specs_path = f"{path}specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_path, path + "result/")
        app.run()

        shutil.rmtree(path)
