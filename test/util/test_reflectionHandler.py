from unittest import TestCase

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.KmerHelper import KmerHelper
from source.util.ReflectionHandler import ReflectionHandler


class TestReflectionHandler(TestCase):
    def test_get_class_from_path(self):

        filepath = EnvironmentSettings.root_path + "/source/util/KmerHelper.py"

        cls = ReflectionHandler.get_class_from_path(filepath, "KmerHelper")
        self.assertEqual(KmerHelper, cls)

        cls = ReflectionHandler.get_class_from_path(filepath)
        self.assertEqual(KmerHelper, cls)

    def test_get_class_by_name(self):
        cls = ReflectionHandler.get_class_by_name("KmerHelper")
        self.assertEqual(KmerHelper, cls)

    def test_exists(self):
        self.assertTrue(ReflectionHandler.exists("ReflectionHandler"))
        self.assertFalse(ReflectionHandler.exists("RandomClassName"))
