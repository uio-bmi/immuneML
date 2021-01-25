import shutil
from unittest import TestCase

import yaml

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestDefaultParamsLoader(TestCase):
    def test_load(self):
        params = {
            "a": 1,
            "b": True
        }

        path = EnvironmentSettings.tmp_test_path / "defaultparamsloader/"
        PathBuilder.build(path)

        with open(path / "mixcr_params.yaml", "w") as file:
            yaml.dump(params, file)

        loaded = DefaultParamsLoader.load(path, "MiXCR")

        self.assertTrue(all(key in loaded.keys() for key in params.keys()))
        self.assertEqual(1, loaded["a"])
        self.assertEqual(True, loaded["b"])
        self.assertEqual(2, len(loaded.keys()))

        shutil.rmtree(path)
