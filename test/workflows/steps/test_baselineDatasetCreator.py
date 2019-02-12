from unittest import TestCase

from source.workflows.steps.BaselineDatasetCreator import BaselineDatasetCreator


class TestBaselineDatasetCreator(TestCase):
    def test_check_prerequisites(self):
        self.assertRaises(AssertionError, BaselineDatasetCreator.check_prerequisites, {})
        self.assertRaises(AssertionError, BaselineDatasetCreator.check_prerequisites, None)
        self.assertRaises(AssertionError, BaselineDatasetCreator.check_prerequisites, {"result_path": None})
        self.assertRaises(AssertionError, BaselineDatasetCreator.check_prerequisites, {"data_loader": None})

    def test_perform_step(self):
        self.fail()
