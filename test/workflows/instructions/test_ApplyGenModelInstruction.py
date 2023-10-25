from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestApplyGenModelInstruction(TestCase):
    def test_run(self):

        path = EnvironmentSettings.tmp_test_path / "applygenmodeltest/"
        PathBuilder.build(path)
        self.assertEqual(True, False)  # add assertion here

