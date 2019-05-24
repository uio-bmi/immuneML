import pickle
import shutil
from unittest import TestCase

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.Dataset import Dataset
from source.environment.EnvironmentSettings import EnvironmentSettings


class TestPickleExporter(TestCase):
    def test_export(self):
        dataset = Dataset(filenames=["f1.pkl", "f2.pkl"])
        PickleExporter.export(dataset, EnvironmentSettings.root_path + "test/tmp/", "dataset.pkl")

        with open(EnvironmentSettings.root_path + "test/tmp/dataset.pkl", "rb") as file:
            dataset2 = pickle.load(file)

        shutil.rmtree(EnvironmentSettings.root_path + "test/tmp/")

        self.assertTrue(isinstance(dataset2, Dataset))
        self.assertEqual(2, len(dataset2.get_filenames()))
        self.assertEqual("f1.pkl", dataset2.get_filenames()[0])
