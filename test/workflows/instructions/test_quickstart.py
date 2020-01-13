import os
import shutil
from unittest import TestCase

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.quickstart import Quickstart


class TestQuickstart(TestCase):
    def test(self):
        path = EnvironmentSettings.tmp_test_path + "quickstart/"
        PathBuilder.build(path)

        quickstart = Quickstart()
        quickstart.run(path)

        self.assertTrue(os.path.isfile(path+"full_specs.yaml"))
        self.assertTrue(os.path.isfile(path+"assessment_RANDOM/all_predictions.csv"))

        shutil.rmtree(path)


