import shutil
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestDatasetExportHTMLOutput(TestCase):

    def test(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "integration_dataset_gen_html/")
        dataset_path = path / "initial_dataset/"

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
                            "result_path": str(dataset_path)
                        }
                    }
                }
            },
            "instructions": {"instr1": {"type": "DatasetExport", "export_formats": ["Pickle", "AIRR"], "datasets": ["d1"]}},
            "output": {"format": "HTML"}
        }

        specs_path = path / "specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_path, path / "result/")
        app.run()

        shutil.rmtree(path)
