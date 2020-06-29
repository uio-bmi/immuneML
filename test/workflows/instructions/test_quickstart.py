import os
import shutil
from glob import glob
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

        self.assertTrue(os.path.isfile(path + "quickstart/full_specs.yaml"))
        self.assertEqual(2, len(glob(path + "quickstart/assessment_random/split_1/**/test_predictions.csv", recursive=True)))
        self.assertTrue(os.path.isfile(glob(path + "quickstart/assessment_random/split_1/**/test_predictions.csv", recursive=True)[0]))

        shutil.rmtree(path, ignore_errors=True)
