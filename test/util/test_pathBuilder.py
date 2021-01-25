import os
import shutil
from unittest import TestCase

from immuneML.util.PathBuilder import PathBuilder


class TestPathBuilder(TestCase):
    def test_build(self):
        path = "testpathbuilder/"
        if not os.path.isdir(path):
            PathBuilder.build(path)
            self.assertTrue(os.path.isdir(path))

        if os.path.isdir(path):
            PathBuilder.build(path)
            self.assertTrue(os.path.isdir(path))

        shutil.rmtree(path)
