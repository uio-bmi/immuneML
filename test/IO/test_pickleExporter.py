import pickle
import shutil
from unittest import TestCase

from source.IO.PickleExporter import PickleExporter
from source.data_model.dataset.Dataset import Dataset


class TestPickleExporter(TestCase):
    def test_export(self):
        dataset = Dataset(filenames=["f1.pkl", "f2.pkl"])
        PickleExporter.export(dataset, "./tmp/", "dataset.pkl")

        with open("./tmp/dataset.pkl", "rb") as file:
            dataset2 = pickle.load(file)

        shutil.rmtree("./tmp/")

        self.assertTrue(isinstance(dataset2, Dataset))
        self.assertEqual(2, len(dataset2.filenames))
        self.assertEqual("f1.pkl", dataset2.filenames[0])
