from unittest import TestCase

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.signal_implanting.FullSequenceImplanting import FullSequenceImplanting
from immuneML.simulation.signal_implanting.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting.ReceptorImplanting import ReceptorImplanting
from immuneML.util.KmerHelper import KmerHelper
from immuneML.util.ReflectionHandler import ReflectionHandler


class TestReflectionHandler(TestCase):
    def test_get_class_from_path(self):

        filepath = EnvironmentSettings.root_path / "/immuneML/util/KmerHelper.py"

        cls = ReflectionHandler.get_class_from_path(filepath, "KmerHelper")
        self.assertEqual(KmerHelper, cls)

        cls = ReflectionHandler.get_class_from_path(filepath)
        self.assertEqual(KmerHelper, cls)

    def test_get_class_by_name(self):
        cls = ReflectionHandler.get_class_by_name("KmerHelper", "util")
        self.assertEqual(KmerHelper, cls)

    def test_exists(self):
        self.assertTrue(ReflectionHandler.exists("ReflectionHandler", "util"))
        self.assertTrue(ReflectionHandler.exists("ReflectionHandler"))
        self.assertFalse(ReflectionHandler.exists("RandomClassName"))

    def test_discover_classes_by_partial_name(self):
        classes = ReflectionHandler.discover_classes_by_partial_name("Implanting", "simulation/signal_implanting_strategy/")
        self.assertListEqual(sorted(['HealthySequenceImplanting', 'ReceptorImplanting', 'FullSequenceImplanting']), sorted(classes))

    def test_get_classes_by_partial_name(self):
        classes = ReflectionHandler.get_classes_by_partial_name("Implanting", "simulation/signal_implanting_strategy/")
        self.assertSetEqual({HealthySequenceImplanting, ReceptorImplanting, FullSequenceImplanting}, set(classes))