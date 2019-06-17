import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.dsl.AssessmentType import AssessmentType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataSplitter import DataSplitter


class TestDataSplitter(TestCase):

    def test_perform_step(self):
        dataset = Dataset(filenames=["file1.pkl", "file2.pkl", "file3.pkl", "file4.pkl", "file5.pkl", "file6.pkl", "file7.pkl", "file8.pkl"])
        training_percentage = 0.7

        trains, tests = DataSplitter.perform_step({
            "dataset": dataset,
            "training_percentage": training_percentage,
            "assessment_type": "random",
            "count": 5
        })

        self.assertTrue(isinstance(trains[0], Dataset))
        self.assertTrue(isinstance(tests[0], Dataset))
        self.assertEqual(len(trains[0].get_filenames()), 5)
        self.assertEqual(len(tests[0].get_filenames()), 3)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))
        self.assertEqual(5, len(np.unique(trains[0].get_filenames())))

        trains, tests = DataSplitter.perform_step({
            "dataset": dataset,
            "assessment_type": "loocv",
        })

        self.assertTrue(isinstance(trains[0], Dataset))
        self.assertTrue(isinstance(tests[0], Dataset))
        self.assertEqual(len(trains[0].get_filenames()), 7)
        self.assertEqual(len(tests[0].get_filenames()), 1)
        self.assertEqual(8, len(trains))
        self.assertEqual(8, len(tests))

        trains, tests = DataSplitter.perform_step({
            "dataset": dataset,
            "assessment_type": "k_fold_cv",
            "count": 5
        })

        self.assertTrue(isinstance(trains[0], Dataset))
        self.assertTrue(isinstance(tests[0], Dataset))
        self.assertEqual(len(trains[0].get_filenames()), 6)
        self.assertEqual(len(tests[0].get_filenames()), 2)
        self.assertEqual(5, len(trains))
        self.assertEqual(5, len(tests))

    def test_build_new_metadata(self):

        path = EnvironmentSettings.tmp_test_path + "data_splitter/"
        PathBuilder.build(path)

        df = pd.DataFrame(data={"key1": [0, 1, 2, 3, 4, 5], "key2": [0, 1, 2, 3, 4, 5]})
        df.to_csv(path+"metadata.csv")

        filepath = DataSplitter.build_new_metadata(path+"metadata.csv", [1, 3, 4], AssessmentType.k_fold, 2, DataSplitter.TRAIN)

        df2 = pd.read_csv(filepath, index_col=0)
        self.assertEqual(3, df2.shape[0])
        self.assertEqual(1, df2.iloc[0, 0])
        self.assertEqual(3, df2.iloc[1, 1])
        self.assertEqual(4, df2.iloc[2, 0])

        shutil.rmtree(path)
