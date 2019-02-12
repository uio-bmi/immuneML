import os
import pickle
import shutil
from unittest import TestCase

from source.IO.PickleLoader import PickleLoader
from source.data_model.dataset.Dataset import Dataset
from source.util.PathBuilder import PathBuilder


class TestPickleLoader(TestCase):
    def test_load(self):
        dataset = Dataset(filenames=["f1.pkl", "f2.pkl"])

        PathBuilder.build("./tmp/")

        with open("./tmp/dataset.pkl", "wb") as file:
            pickle.dump(dataset, file)

        dataset2 = PickleLoader.load("./tmp/dataset.pkl")

        shutil.rmtree("./tmp/")

        self.assertEqual(2, len(dataset2.filenames))
        self.assertEqual("f2.pkl", dataset2.filenames[1])
