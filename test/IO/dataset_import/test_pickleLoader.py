import pickle
import shutil
from unittest import TestCase

from source.IO.dataset_import.PickleLoader import PickleLoader
from source.data_model.dataset.Dataset import Dataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestPickleLoader(TestCase):
    def test_load(self):
        dataset = Dataset(filenames=["f1.pkl", "f2.pkl"])
        path = EnvironmentSettings.root_path + "test/tmp/pathbuilder/"
        PathBuilder.build(path)

        with open(path + "dataset.pkl", "wb") as file:
            pickle.dump(dataset, file)

        dataset2 = PickleLoader.load(path + "dataset.pkl")

        shutil.rmtree(path)

        self.assertEqual(2, len(dataset2.get_filenames()))
        self.assertEqual("f2.pkl", dataset2.get_filenames()[1])
