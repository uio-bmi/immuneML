from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.workflows.steps.DataSplitter import DataSplitter


class TestDataSplitter(TestCase):

    def test_perform_step(self):
        dataset = Dataset()
        dataset.filenames = ["file1.pkl", "file2.pkl", "file3.pkl", "file4.pkl", "file5.pkl", "file6.pkl", "file7.pkl", "file8.pkl"]
        training_percentage = 0.7

        trains, tests = DataSplitter.perform_step({
            "dataset": dataset,
            "training_percentage": training_percentage,
            "assessment_type": "random",
            "count": 5
        })

        self.assertTrue(isinstance(trains[0], Dataset))
        self.assertTrue(isinstance(tests[0], Dataset))
        self.assertEqual(len(trains[0].filenames), 5)
        self.assertEqual(len(tests[0].filenames), 3)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))

        trains, tests = DataSplitter.perform_step({
            "dataset": dataset,
            "assessment_type": "loocv",
        })

        self.assertTrue(isinstance(trains[0], Dataset))
        self.assertTrue(isinstance(tests[0], Dataset))
        self.assertEqual(len(trains[0].filenames), 7)
        self.assertEqual(len(tests[0].filenames), 1)
        self.assertEqual(8, len(trains))
        self.assertEqual(8, len(tests))

        trains, tests = DataSplitter.perform_step({
            "dataset": dataset,
            "assessment_type": "k_fold_cv",
            "count": 5
        })

        self.assertTrue(isinstance(trains[0], Dataset))
        self.assertTrue(isinstance(tests[0], Dataset))
        self.assertEqual(len(trains[0].filenames), 6)
        self.assertEqual(len(tests[0].filenames), 2)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))
