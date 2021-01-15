import os
import shutil
from unittest import TestCase

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


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
                            "repertoire_count": 50,
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
                    "subsampled_dataset_sizes": [20, 30],
                    "dataset_export_formats": ["Pickle", 'AIRR']
                }
            }
        }

    def test_subsampling(self):
        import faulthandler
        faulthandler.enable()

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "subsampling_workflow/")
        repertoire_specs = self.build_specs(path)

        specs_filename = path / "specs.yaml"
        with open(specs_filename, "w") as file:
            yaml.dump(repertoire_specs, file)

        app = ImmuneMLApp(specs_filename, path / "result/")
        app.run()

        shutil.rmtree(path)
