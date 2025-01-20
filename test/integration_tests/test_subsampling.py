import os
import shutil
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestSubsamplingWorkflow(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def build_specs(self, path) -> dict:
        return {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 5,
                            "result_path": str(path),
                            "labels": {
                                "cmv": {
                                    True: 0.5,
                                    False: 0.5
                                }
                            }
                        }
                    }
                },
            },
            "instructions": {
                "subsampling": {
                    "type": "Subsampling",
                    "dataset": "d1",
                    "subsampled_dataset_sizes": [2, 3],
                    "dataset_export_formats": ['AIRR']
                }
            }
        }

    def test_subsampling(self):

        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "subsampling_workflow/")
        repertoire_specs = self.build_specs(path)

        specs_filename = path / "specs.yaml"
        with open(specs_filename, "w") as file:
            yaml.dump(repertoire_specs, file)

        app = ImmuneMLApp(specs_filename, path / "result/")
        app.run()

        shutil.rmtree(path)
