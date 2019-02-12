from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.workflows.steps.DataSplitter import DataSplitter


class TestDataSplitter(TestCase):
    def test_check_prerequisites(self):
        self.assertRaises(AssertionError, DataSplitter.check_prerequisites, {})
        self.assertRaises(AssertionError, DataSplitter.check_prerequisites, None)
        self.assertRaises(AssertionError, DataSplitter.check_prerequisites, {"dataset": None})
        self.assertRaises(AssertionError, DataSplitter.check_prerequisites, {"training_percentage": Dataset()})

    def test_perform_step(self):
        dataset = Dataset()
        dataset.filenames = ["file1.pkl", "file2.pkl", "file3.pkl", "file4.pkl", "file5.pkl", "file6.pkl", "file7.pkl", "file8.pkl"]
        training_percentage = 0.7

        train, test = DataSplitter.perform_step({
            "dataset": dataset,
            "training_percentage": training_percentage
        })

        self.assertTrue(isinstance(train, Dataset))
        self.assertTrue(isinstance(test, Dataset))
        self.assertEqual(len(train.filenames), 5)
        self.assertEqual(len(test.filenames), 3)
